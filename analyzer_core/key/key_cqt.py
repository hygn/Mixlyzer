import numpy as np
import librosa


class keydetect:
    """
    CQT-based key detector with note-gated chroma and optional
    relative-minor inference to reduce template scoring by half.

    - Input: mono ndarray only (N,) or (N,1) with sample rate
    - Computes high-resolution CQT, gates to semitone centers
    - Aggregates to a 12-bin pitch-class profile
    - Scores 12 major keys; optionally derives relative-minor candidates
    - Can weight scores using I/IV/V (or i/iv/v) chord presence
    """

    def __init__(self, y_mono: np.ndarray, sr: int):
        """Initialize with a mono audio ndarray and sample rate.
        y_mono: shape (N,) or (N,1). Raises if multi-channel.
        """
        if not isinstance(y_mono, np.ndarray):
            raise TypeError("y_mono must be a numpy.ndarray")
        if y_mono.ndim == 1:
            y = y_mono
        elif y_mono.ndim == 2 and y_mono.shape[1] == 1:
            y = y_mono[:, 0]
        else:
            raise ValueError("key_cqt expects mono ndarray (N,) or (N,1)")
        self.mono = y.astype(np.float32, copy=False)
        self.samprate = int(sr)

    def _pc_index(self):
        return {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
            'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }

    def _build_major_only(self):
        """Return major key info and templates only (12), plus mapping for minors.
        Returns
        -------
        major_names : list[str]
        minor_names : list[str]   # aligned: minor_names[i] is relative minor of major_names[i]
        camelot     : dict[str,str]
        maj_templates : np.ndarray, shape (12,12)
        major_roots_pc : list[int]
        pc_index    : dict[str,int]
        """
        pc_index = self._pc_index()

        # Conventions consistent with earlier code
        major_names = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
        minor_names = ["A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D"]

        camelot = {
            # Major (B ring)
            "C:major": "8B",  "G:major": "9B",  "D:major": "10B", "A:major": "11B",
            "E:major": "12B", "B:major": "1B",  "F#:major": "2B", "Db:major": "3B",
            "Ab:major": "4B", "Eb:major": "5B", "Bb:major": "6B",  "F:major": "7B",
            # Minor (A ring)
            "A:minor": "8A",  "E:minor": "9A",  "B:minor": "10A", "F#:minor": "11A",
            "Db:minor": "12A", "Ab:minor": "1A",  "Eb:minor": "2A",  "Bb:minor": "3A",
            "F:minor": "4A",  "C:minor": "5A",  "G:minor": "6A",  "D:minor": "7A",
        }

        major_roots_pc = [pc_index[n] for n in major_names]

        maj_pattern = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        maj_templates = np.stack([np.roll(maj_pattern, r) for r in major_roots_pc], axis=0)

        return major_names, minor_names, camelot, maj_templates, major_roots_pc, pc_index

    def _compute_chroma_note_gated(self,
                                   y: np.ndarray,
                                   sr: int,
                                   hop_length: int = 512,
                                   bins_per_octave: int = 36,
                                   n_octaves: int | None = 7,
                                   fmin: float | None = None,
                                   note_tolerance_cents: float = 20.0,
                                   allowed_pitch_classes=None) -> np.ndarray:
        """
        Compute a 12-bin chromagram from high-resolution CQT, but only
        accumulate energy from CQT bins whose center frequency is within
        `note_tolerance_cents` of the nearest equal-tempered semitone.
        """
        if fmin is None:
            fmin = float(librosa.note_to_hz('C1'))
        if n_octaves is None:
            n_octaves = 7
        n_bins = int(n_octaves * bins_per_octave)

        C = librosa.cqt(y=y, sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave,
                        n_bins=n_bins, fmin=fmin)
        A = np.abs(C).astype(np.float32)
        if A.ndim != 2:
            return np.zeros((12, 0), dtype=np.float32)
        T = A.shape[1]

        freqs = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
        midi = librosa.hz_to_midi(freqs)
        nearest = np.rint(midi)
        cents = 100.0 * np.abs(midi - nearest)
        pc = (nearest.astype(int) % 12)

        # Allowed pitch classes mapping
        if allowed_pitch_classes is not None:
            if isinstance(allowed_pitch_classes, (list, tuple, set, np.ndarray)):
                allowed_list = list(allowed_pitch_classes)
            else:
                allowed_list = [allowed_pitch_classes]
            pc_map = self._pc_index()
            pcs = []
            for v in allowed_list:
                if isinstance(v, (int, np.integer)):
                    pcs.append(int(v) % 12)
                elif isinstance(v, str):
                    k = v.strip().replace('♯', '#').replace('♭', 'b')
                    pcs.append(pc_map[k])
                else:
                    raise ValueError(f"Invalid pitch class spec: {v!r}")
            allowed_set = set(pcs)
        else:
            allowed_set = None

        mask_bins = cents <= float(note_tolerance_cents)
        if allowed_set is not None:
            mask_bins = mask_bins & np.isin(pc, list(allowed_set))

        chroma12 = np.zeros((12, T), dtype=np.float32)
        for p in range(12):
            idx = np.where(mask_bins & (pc == p))[0]
            if idx.size > 0:
                chroma12[p] = A[idx, :].sum(axis=0)

        return chroma12

    def _chord_counts(self, chroma12: np.ndarray, root_pc: int, is_major: bool):
        """Compute chord-presence strength for I/IV/V (or i/iv/v) across frames.

        This replaces hard threshold counting with a continuous chord strength:
        for each frame, take the L2 norm of the triad note energies (after
        per-frame max normalization) and sum across frames.

        Parameters
        ----------
        chroma12:
            (12, T) nonnegative chroma.
        root_pc:
            Tonic pitch class (0-11).
        is_major:
            Whether to evaluate major-mode functions (I/IV/V) or minor (i/iv/v).
        Returns
        -------
        (strengths_dict, weighted_strength)
        """
        if chroma12.ndim != 2 or chroma12.shape[0] != 12:
            return ({}, 0.0)
        V = chroma12.astype(np.float32, copy=False)
        T = V.shape[1]
        if T == 0:
            return ({}, 0.0)
        frame_max = V.max(axis=0)
        frame_max[frame_max < 1e-12] = 1.0
        VN = V / frame_max

        def triad_strength(root: int, minor: bool) -> float:
            if minor:
                notes = [root % 12, (root + 3) % 12, (root + 7) % 12]
            else:
                notes = [root % 12, (root + 4) % 12, (root + 7) % 12]
            chord_vec = VN[notes, :]  # (3, T)
            per_frame = np.linalg.norm(chord_vec, axis=0) / np.sqrt(3.0)  # [0,1] approx
            return float(np.sum(per_frame))

        if is_major:
            s_I = triad_strength(root_pc, False)
            s_IV = triad_strength((root_pc + 5) % 12, False)
            s_V = triad_strength((root_pc + 7) % 12, False)
            counts = {"I": s_I, "IV": s_IV, "V": s_V}
        else:
            s_i = triad_strength(root_pc, True)
            s_iv = triad_strength((root_pc + 5) % 12, True)
            s_v = triad_strength((root_pc + 7) % 12, True)
            counts = {"i": s_i, "iv": s_iv, "v": s_v}

        weighted = (counts.get("I", 0.0) + counts.get("i", 0.0)) * 1.0
        weighted += (counts.get("IV", 0.0) + counts.get("V", 0.0)) * 0.2
        weighted += (counts.get("iv", 0.0) + counts.get("v", 0.0)) * 0.2
        return counts, float(weighted)

    def keydetect(self, songrange=None, beat_times=None, verbose: bool = True,
                  hop_length: int = 512, bins_per_octave: int = 36,
                  n_octaves: int | None = 4, aggregate: str = "median",
                  note_tolerance_cents: float = 10.0, allowed_pitch_classes=None,
                  fmin_note: str = 'C4',
                  use_chord_weight: bool = True, chord_scale: float = 100.0):
        """
        Run CQT-based key detection on the whole track or a specified range.

        - If relative_minor_infer is True (default), only 12 major templates
          are scored; relative minors are derived per-major using chord cues.
        """
        sr = self.samprate
        seg_start_sec = 0.0
        if songrange is None:
            y = self.mono
        else:
            s0 = max(0, int(songrange[0] * sr))
            s1 = min(len(self.mono), int(songrange[1] * sr))
            seg_start_sec = s0 / float(sr)
            y = self.mono[s0:s1]

        if verbose:
            dur = len(y) / sr
            print(f"[key_cqt] Input mono array | sr={sr}Hz | samples={len(y)} | dur={dur:.2f}s | songrange={songrange}")
            print("[key_cqt] Calculating gated CQT chroma ...")

        chroma = self._compute_chroma_note_gated(
            y, sr,
            hop_length=hop_length,
            bins_per_octave=bins_per_octave,
            n_octaves=n_octaves,
            fmin=float(librosa.note_to_hz(fmin_note)),
            note_tolerance_cents=note_tolerance_cents,
            allowed_pitch_classes=allowed_pitch_classes,
        )

        agg_func = np.median if aggregate == "median" else np.mean
        if chroma.shape[1] > 0 and beat_times is not None:
            bt = np.asarray(beat_times, dtype=float).ravel()
            seg_duration = len(y) / float(sr) if len(y) else 0.0
            seg_end_sec = seg_start_sec + seg_duration
            mask = (bt >= seg_start_sec) & (bt <= seg_end_sec)
            beat_local = bt[mask] - seg_start_sec
            beat_frames = librosa.time_to_frames(beat_local, sr=sr, hop_length=hop_length)
            beat_frames = np.unique(beat_frames[(beat_frames >= 0) & (beat_frames < chroma.shape[1])])
            if beat_frames.size > 0:
                chroma = librosa.util.sync(chroma, beat_frames, aggregate=agg_func)
                if verbose:
                    print(f"[key_cqt] Beat-sync chroma: frames={chroma.shape[1]} | beats_local={beat_local.size}")
            elif verbose:
                print("[key_cqt] No valid beat frames for segment; using raw chroma.")
        elif verbose and chroma.shape[1] > 0:
            print("[key_cqt] Beat-sync skipped; using raw chroma.")

        pc_profile = agg_func(chroma, axis=1) if chroma.shape[1] > 0 else np.zeros(12, dtype=float)
        pc_profile = np.asarray(pc_profile, dtype=float)
        if pc_profile.sum() > 0:
            pc_profile = pc_profile / (pc_profile.sum() + 1e-12)

        major_names, minor_names, camelot, maj_templates, major_roots_pc, pc_index = self._build_major_only()

        # Baseline scores for 12 majors only
        temp_major = maj_templates * 1.0 + 0.5
        scores_major = temp_major @ pc_profile  # (12,)
        scores_major = np.asarray(scores_major, dtype=float)

        # Build candidates: per major root, make (major, relative minor)
        candidates = []  # list of dicts {name, score, chords}
        for i, root_pc in enumerate(major_roots_pc):
            base = float(scores_major[i])

            # Major side
            chords_mj, w_mj = ({}, 0.0)
            if chroma.shape[1] > 0 and use_chord_weight:
                chords_mj, w_mj = self._chord_counts(chroma, root_pc, True)
            score_mj = base * ((w_mj / float(chord_scale)) + 1.0) if use_chord_weight else base
            name_mj = f"{major_names[i]}:major"
            candidates.append({
                "key": name_mj,
                "camelot": camelot.get(name_mj, ""),
                "score": score_mj,
                "chords": chords_mj if use_chord_weight else {},
            })

            # Relative minor side: 9 semitones up from major root
            rel_min_name = minor_names[i]
            rel_min_pc = pc_index[rel_min_name]
            chords_mn, w_mn = ({}, 0.0)
            if chroma.shape[1] > 0 and use_chord_weight:
                chords_mn, w_mn = self._chord_counts(chroma, rel_min_pc, False)
            score_mn = base * ((w_mn / float(chord_scale)) + 1.0) if use_chord_weight else base
            name_mn = f"{rel_min_name}:minor"
            candidates.append({
                "key": name_mn,
                "camelot": camelot.get(name_mn, ""),
                "score": score_mn,
                "chords": chords_mn if use_chord_weight else {},
            })

        # Sort all candidates
        candidates.sort(key=lambda d: d["score"], reverse=True)

        if verbose:
            print("Top key candidates (major+relative minor inference):")
            for i in range(min(5, len(candidates))):
                nm = candidates[i]["key"]
                cam = candidates[i]["camelot"]
                sc = candidates[i]["score"]
                print(f"[key_cqt]   {nm:>8} ({cam:>3}) | score: {sc:.4f}")

        ret = {"keys": [], "notes": []}
        for c in candidates:
            ret["keys"].append({
                "key": c["key"],
                "camelot": c["camelot"],
                "score": float(c["score"]),
                "chords": c["chords"],
            })

        note_names_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for i, val in enumerate(pc_profile):
            ret["notes"].append({
                "note": note_names_sharp[i],
                "count": float(val),
                "relerror": 0.0,
            })

        return ret
