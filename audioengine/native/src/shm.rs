//! Shared memory made by the GUI process (Python's multiprocessing.shared_memory), opened here by its name:
//! the control block (commands in, state out) and each track's PCM.

pub struct Shm {
    ptr: *mut u8,
    len: usize,
    #[cfg(windows)]
    handle: windows_sys::Win32::Foundation::HANDLE,
}

// The mapping is plain memory; who reads and writes which part is the protocol's business (ipc.rs).
unsafe impl Send for Shm {}
unsafe impl Sync for Shm {}

impl Shm {
    /// Opens the block Python created as SharedMemory(name), ``len`` bytes of it.
    #[cfg(windows)]
    pub fn open(name: &str, len: usize, writable: bool) -> Result<Shm, String> {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, OpenFileMappingW,
        };
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        let access = if writable { FILE_MAP_READ | FILE_MAP_WRITE } else { FILE_MAP_READ };
        let handle = unsafe { OpenFileMappingW(access, 0, wide.as_ptr()) };
        if handle.is_null() {
            return Err(format!("OpenFileMapping({name}) failed: {}", std::io::Error::last_os_error()));
        }
        // a view larger than the section is refused here (MapViewOfFile fails), so ``len`` bytes are there
        let view = unsafe { MapViewOfFile(handle, access, 0, 0, len) };
        if view.Value.is_null() {
            let err = std::io::Error::last_os_error();
            unsafe { CloseHandle(handle) };
            return Err(format!("MapViewOfFile({name}) failed: {err}"));
        }
        Ok(Shm { ptr: view.Value as *mut u8, len, handle })
    }

    #[cfg(unix)]
    pub fn open(name: &str, len: usize, writable: bool) -> Result<Shm, String> {
        let path = std::ffi::CString::new(format!("/{}", name.trim_start_matches('/'))).map_err(|e| e.to_string())?;
        let flags = if writable { libc::O_RDWR } else { libc::O_RDONLY };
        let fd = unsafe { libc::shm_open(path.as_ptr(), flags, 0o600) };
        if fd < 0 {
            return Err(format!("shm_open({name}) failed: {}", std::io::Error::last_os_error()));
        }
        // the block holds the bytes asked for (a mapping past its end would fault when read: SIGBUS)
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut stat) } != 0 || (stat.st_size as u64) < len as u64 {
            unsafe { libc::close(fd) };
            return Err(format!("shm {name}: smaller than the {len} bytes asked for"));
        }
        let prot = if writable { libc::PROT_READ | libc::PROT_WRITE } else { libc::PROT_READ };
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len.max(1), prot, libc::MAP_SHARED, fd, 0) };
        unsafe { libc::close(fd) };   // the mapping stays
        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap({name}) failed: {}", std::io::Error::last_os_error()));
        }
        Ok(Shm { ptr: ptr as *mut u8, len })
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for Shm {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.ptr as *mut _ });
            CloseHandle(self.handle);
        }
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr as *mut _, self.len.max(1));
        }
    }
}

/// A track's PCM: [frames, channels] float32, interleaved (NumPy's C order), read only.
pub struct Pcm {
    shm: Backing,
    pub frames: usize,
    pub channels: usize,
}

/// Where a PCM's samples are: the GUI's shared block (the tests': their own).
enum Backing {
    Shared(Shm),
    #[cfg(test)]
    Owned(Vec<f32>),
}

impl Pcm {
    pub fn open(name: &str, frames: usize, channels: usize) -> Result<Pcm, String> {
        let bytes = frames
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| format!("PCM {name}: {frames} frames of {channels} channels do not fit in memory"))?;
        let shm = Shm::open(name, bytes, false)?;
        Ok(Pcm { shm: Backing::Shared(shm), frames, channels })
    }

    /// A PCM of the test's own (interleaved).
    #[cfg(test)]
    pub fn owned(data: Vec<f32>, channels: usize) -> Pcm {
        Pcm { frames: data.len() / channels, channels, shm: Backing::Owned(data) }
    }

    pub fn data(&self) -> &[f32] {
        match &self.shm {
            Backing::Shared(shm) => unsafe {
                std::slice::from_raw_parts(shm.ptr() as *const f32, self.frames * self.channels)
            },
            #[cfg(test)]
            Backing::Owned(v) => v,
        }
    }
}
