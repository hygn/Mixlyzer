"""Every operation on the library, by name: what the rest of the app asks the library's owner to do.

A job is ``job(lib, *args)``, run on the owner's thread (core.library.service: `Library` is its connection, its
folder and its feature store). The app never calls a job itself: it asks the owner,

    library.read(jobs.track_tieups, uid, done=show)          # the GUI: told on its thread
    library.write(jobs.rename_playlist, uid, name, done=...)  # a write: the snapshot read again before done
    library.call(jobs.song_of, uid)                           # another thread of the app: waits for it

so this module and core.analysis_library.jobs (the tracks' stored analyses, {uid}.npz) are the library's whole
surface: below them (core.library.db, tieups, variants, music_index, thumbnails; core.analysis_library.store) is how
the library is kept, which nothing outside them touches.

A write job is marked with the snapshot's parts it changes (``@changes(TRACKS, READINGS)``; ``@changes()``: none),
so the owner reads only those again; a write job left unmarked has the snapshot read whole after it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional

from PySide6 import QtCore, QtGui

from core.library import mixability, music_index, package, thumbnails, tieups, variants
from core.library.records import READING_FIELDS, IndexTrack, PlaylistTree, TrackRow
from core.library.segments import track_index
from core.library.service import PLAYLISTS, READINGS, TIEUPS, TRACKS, VARIANTS, changes
from core.search import mixability as mix_search
from core.search.variants import groups as variant_groups
from core.tieups.domain import Catalog, TieUp, Work
from analyzer.search import mixability as mix_fp
from analyzer.search import variants as variant_fp
from core.analysis_library import jobs as analysis_jobs
from core.analysis_library.stored import StoredAnalysis, TrackEdits, apply_edits, index_marks


# ---- tracks ----

def track(lib, uid: str) -> Optional[TrackRow]:
    return lib.db.get_by_uid(uid)


def all_tracks(lib) -> list[TrackRow]:
    return lib.db.list_all()


@changes(TRACKS)
def set_track_path(lib, uid: str, path: str) -> None:
    """The track's file moved: its new path (one file, one track: ValueError when another track has it)."""
    lib.db.set_path(uid, path)


@changes(TRACKS, READINGS)
def save_track(lib, row: TrackRow, path_changed: bool) -> Optional[TrackRow]:
    """The Edit tab's changes: the path first (checked: one file, one track; the file's size and time follow it),
    then the metadata (its readings read again). The row as stored."""
    if path_changed:
        lib.db.set_path(row.uid, row.path)
    lib.db.upsert(replace(row, file_mtime=None, file_size=None))
    return lib.db.get_by_uid(row.uid)


def remove_tracks(lib, uids: list[str]) -> None:
    """Tracks out of the library: their rows (their readings, tie-ups and playlist entries go with them), features,
    thumbnails, and their place in the indexes made from them (each a pass over its index). The rows go first: a
    file that cannot be removed (in use) is left behind, said on the console, and the others still go."""
    lib.db.delete_tracks(uids)
    for uid in uids:
        for what, remove, args in (("analysis", lib.features.delete, (uid,)),
                                   ("thumbnail", thumbnails.delete_thumbnail, (lib.path, uid))):
            try:
                remove(*args)
            except OSError as exc:
                print(f"[Library] a removed track's {what} left behind: {exc}")
    try:
        drop_from_indexes(lib, uids)
    except Exception as exc:        # left for Index Now to drop
        print(f"[Library] removed tracks left in the indexes: {exc}")


def save_edits(lib, uid: str, edits: TrackEdits, sync: bool = True) -> str:
    """The editor's save: ``edits`` into the track's stored analysis (each mark's block replaced whole) and, with
    ``sync`` (Settings > Library, libconfig.sync_stored_analysis), the track's rows in library.db made from them
    (track_index: the BPM and key the library shows and the Export tab's tags take, the tables the phrase /
    transition filters read; without, library.db is left for Sync Now, sync_index). Returns the uid. Unmarked (the
    snapshot read whole, sig_lib_updated sent): the filters run again on it. One transaction, the NPZ saved last
    inside it: a failure anywhere rolls library.db back whole, so it never holds half an edit."""
    stored = lib.features.load(uid)
    apply_edits(stored, edits)
    with lib.db.tx():
        if sync:
            lib.db.write_track_index(uid, track_index(*index_marks(stored)))
        lib.features.save(uid, stored)
    # the mix candidates' bars follow the downbeats as edited (the beats' features read again only where the beats
    # changed: Index Now); a failure leaves the bars as they were, never the save undone
    try:
        mix_bars_again(lib, uid, StoredAnalysis.from_arrays(stored))
    except Exception as exc:
        print(f"[Mix candidates] bars not made again: {exc}")
    return uid


# tracks a sync job reads (sync_index): about 1 ms a track measured, so the GUI's jobs wait well under a second
SYNC_BATCH = 50


@dataclass(frozen=True)
class IndexSync:
    """What a sync did: the tracks whose rows in library.db changed, and those without a stored analysis that reads
    (left as they are)."""
    changed: int = 0
    unread: int = 0


@changes()
def sync_index(lib, uids: list[str]) -> IndexSync:
    """library.db made to agree with the tracks' stored analyses (Settings > Library's Sync Now, SYNC_BATCH tracks a
    job): each track's rows written from its stored marks (track_index, as the editor's save writes them; only where
    they differ), the BPM and key typed in the Edit Track tab replaced too, the rest of its metadata untouched,
    nothing of the stored analyses written. A stored analysis that does not read (a damaged file, a block of the
    wrong shape) is counted and the others go on. The snapshot is not read again: the run's end does that once,
    index_synced, where a row changed."""
    changed = unread = 0
    for uid in uids:
        try:
            marks = analysis_jobs.stored_marks(lib, uid)
            index = track_index(*marks) if marks is not None else None
        except Exception as exc:
            print(f"[Library] a stored analysis not read for the sync: {exc!r}")
            index = None
        if index is None:
            unread += 1
            continue
        changed += lib.db.update_track_index(uid, index)
    return IndexSync(changed, unread)


@changes(TRACKS)
def index_synced(lib) -> None:
    """Nothing: a sync's end where it changed rows (the snapshot's tracks read again, sig_lib_updated sent, so the
    library's columns and its phrase / transition filters follow)."""


def tag_sources(lib, uid: str) -> dict:
    return lib.db.tag_sources(uid)


@changes()
def set_tag_sources(lib, uid: str, sources: dict) -> None:
    lib.db.set_tag_sources(uid, sources)


@changes(TRACKS)
def touch_file_stats(lib, path: str) -> None:
    """The track's file written (its tags): its new size and time."""
    lib.db.touch_file_stats(path)


def tag_values(lib, uid: str, with_tieups: bool, lang: str = "") -> tuple[Optional[TrackRow], str]:
    """What the Export tab writes into the file's tags: the track's row, and its tie-ups' comment suffix ("" without
    ``with_tieups``; the titles in ``lang``, "": the Tie-up Titles setting's)."""
    from core.tieups.domain import comment_suffix
    row = lib.db.get_by_uid(uid)
    return row, comment_suffix(tieups.track_tieups(lib.db.conn, uid), lang) if with_tieups and row is not None else ""


def song_of(lib, uid: str, untagged: bool = False) -> Optional[tuple[str, str, str]]:
    """The track's (title, artist, album) as the tie-up web lookup asks Wikimedia; None for no track, no title, or
    (``untagged``) a track with tie-ups already."""
    if untagged and tieups.has_tieups(lib.db.conn, uid):
        return None
    row = lib.db.get_by_uid(uid)
    return (row.title, row.artist or "", row.album or "") if row is not None and row.title else None


# ---- readings (the search's: core.search.reading) ----

def readings(lib, uid: str) -> tuple[dict, dict]:
    """The track's readings by field ((reading, source) pairs) and the automatic ones the user removed."""
    return lib.db.readings(uid), lib.db.removed_readings(uid)


@changes(READINGS)
def set_user_readings(lib, uid: str, edits: dict) -> Optional[TrackRow]:
    """The Search Index tab's changes: field -> (the readings the user added, the automatic ones removed)."""
    for field, (added, removed) in edits.items():
        lib.db.set_user_readings(uid, field, added, removed)
    return lib.db.get_by_uid(uid)


@changes(READINGS, TIEUPS)
def rebuild_name_keys(lib) -> int:
    """Every title / artist / album's and tie-up name's 초성 and sort keys made again from its text and readings
    (Settings > Search's Rebuild Sort Index). The names made."""
    made = lib.db.refresh_name_keys(rebuild=True)
    with lib.db.tx():
        made += tieups.refresh_name_keys(lib.db.conn, rebuild=True)
    return made


def reading_texts(lib) -> tuple[list[str], list[str]]:
    """Re-read Library's input: (the tracks' uids, every text read for the search: their titles, artists and albums
    and the tie-up works' names, each once)."""
    rows = lib.db.list_all(order_by="added_ts ASC")
    texts = dict.fromkeys(text or "" for row in rows for text in (row.title, row.artist, row.album))
    texts.update(dict.fromkeys(tieups.work_names(lib.db.conn)))
    return [row.uid for row in rows], list(texts)


@changes(READINGS)
def apply_readings(lib, uids: list[str], read: dict) -> int:
    """Re-read Library's output for some tracks: their automatic readings from ``read`` (text -> readings), the
    user's kept. The fields changed."""
    changed = 0
    for uid in uids:
        before = lib.db.readings(uid)
        lib.db.refresh_readings(uid, read)
        after = lib.db.readings(uid)
        changed += sum(before.get(f) != after.get(f) for f in READING_FIELDS)
    return changed


@changes(TIEUPS)
def apply_work_readings(lib, read: dict) -> None:
    """Re-read Library's output for the tie-up works' names."""
    tieups.refresh_work_readings(lib.db.conn, lib.db.reading_languages, lib.db.han_run, read)


# ---- playlists ----

@changes(PLAYLISTS)
def create_playlist(lib, name: str, folder_uid: Optional[str] = None):
    """A playlist at the end of a folder (None: the top)."""
    return lib.db.create_playlist(name, folder_uid)


@changes(PLAYLISTS)
def rename_playlist(lib, playlist_uid: str, name: str) -> None:
    lib.db.rename_playlist(playlist_uid, name)


@changes(PLAYLISTS)
def delete_playlist(lib, playlist_uid: str) -> None:
    lib.db.delete_playlist(playlist_uid)


@changes(PLAYLISTS)
def create_playlist_folder(lib, name: str, parent_uid: Optional[str] = None):
    """A folder at the end of a folder (None: the top)."""
    return lib.db.create_playlist_folder(name, parent_uid)


@changes(PLAYLISTS)
def rename_playlist_folder(lib, folder_uid: str, name: str) -> None:
    lib.db.rename_playlist_folder(folder_uid, name)


@changes(PLAYLISTS)
def delete_playlist_folder(lib, folder_uid: str) -> None:
    """The folder deleted with its folders and playlists (the tracks stay in the library)."""
    lib.db.delete_playlist_folder(folder_uid)


@changes(PLAYLISTS)
def move_playlist_node(lib, kind: str, uid: str, folder_uid: Optional[str], before=None) -> bool:
    """A playlist or folder (records.PLAYLIST_NODE / FOLDER_NODE) moved into a folder (None: the top), before its child
    ``before`` ((kind, uid); None: at its end); False when nothing moved (a folder into itself is not)."""
    return lib.db.move_playlist_node(kind, uid, folder_uid, before)


@changes(PLAYLISTS)
def move_playlist_node_by(lib, kind: str, uid: str, offset: int) -> bool:
    """A playlist or folder moved ``offset`` places among its folder's children."""
    return lib.db.move_playlist_node_by(kind, uid, offset)


def playlist_tree(lib) -> PlaylistTree:
    """The playlists' folders, the playlists and their entries (the Rekordbox XML's PLAYLISTS)."""
    return PlaylistTree(tuple(lib.db.list_playlist_folders()), tuple(lib.db.list_playlists()),
                        lib.db.playlist_entries())


@changes(PLAYLISTS)
def add_to_playlists(lib, entries: dict[str, list[str]]) -> None:
    """Tracks added at the end of playlists: playlist uid -> track uids."""
    for playlist_uid, uids in entries.items():
        lib.db.add_playlist_tracks(playlist_uid, uids)


@changes(PLAYLISTS)
def move_playlist_entries(lib, playlist_uid: str, positions, before_position) -> bool:
    return lib.db.move_playlist_tracks_before(playlist_uid, positions, before_position)


@changes(PLAYLISTS)
def remove_playlist_entries(lib, playlist_uid: str, positions) -> None:
    lib.db.remove_playlist_tracks(playlist_uid, positions)


# ---- the library's phrase / transition search ----

SEARCHES = ("search_phrase_sections", "search_bpm_transitions", "search_harmonic_key_transitions")


def run_searches(lib, searches: tuple) -> list[set[str]]:
    """The library's filters' searches ((one of SEARCHES, args, keyword args) each): the paths each finds."""
    out = []
    for method, args, kwargs in searches:
        if method not in SEARCHES:
            raise ValueError(f"not a library search: {method}")
        out.append({str(row.path) for row in getattr(lib.db, method)(*args, **dict(kwargs))})
    return out


# ---- tie-ups ----

def track_tieups(lib, uid: str) -> list[TieUp]:
    return tieups.track_tieups(lib.db.conn, uid)


def tieup_catalog(lib) -> Catalog:
    return tieups.catalog(lib.db.conn)


@changes(TIEUPS)
def save_tieup(lib, uid: str, t: TieUp, old: Optional[tuple] = None, versions=()) -> list[str]:
    """A tie-up written as edited (``old``: the key it had), an addition to ``versions`` too. The tracks written."""
    return tieups.save_tieup(lib.db.conn, uid, t, old, versions, lib.db.reading_languages, lib.db.han_run)


@changes(TIEUPS)
def delete_tieup(lib, uid: str, key: tuple) -> list[str]:
    tieups.delete_tieup(lib.db.conn, uid, key)
    return [uid]


@changes(TIEUPS)
def add_found_tieups(lib, tracks: list[str], found: list[TieUp]) -> list[str]:
    """Tie-ups found on the web added to the tracks (core.library.tieups.add_found). The tracks."""
    for t in found:
        for uid in tracks:
            tieups.add_found(lib.db.conn, uid, t, lib.db.reading_languages, lib.db.han_run)
    return list(tracks)


def work_tracks(lib, work_id: str) -> list[str]:
    return tieups.work_tracks(lib.db.conn, work_id)


@changes(TIEUPS)
def save_work(lib, w: Work) -> list[str]:
    """A work written as edited. The tracks using it."""
    tieups.save_work(lib.db.conn, w, lib.db.reading_languages, lib.db.han_run)
    return tieups.work_tracks(lib.db.conn, w.id)


@changes(TIEUPS)
def merge_works(lib, keep: str, drop: str) -> list[str]:
    return tieups.merge_works(lib.db.conn, keep, drop, lib.db.reading_languages, lib.db.han_run)


@changes(TIEUPS)
def delete_work(lib, work_id: str) -> list[str]:
    return tieups.delete_work(lib.db.conn, work_id)


def comment_suffixes(lib, lang: str = "") -> dict[str, str]:
    """Every tagged track's comment suffix, the titles in ``lang`` ("": the Tie-up Titles setting's)."""
    return tieups.comment_suffixes(lib.db.conn, lang)


# ---- the indexes beside the library: the Variants', Search by Music's and the mix candidates' ----

def index_state(lib, uid: str) -> tuple[bool, bool]:
    """Whether the track is in the Variants' index (its current method) and in Search by Music's."""
    return (uid in variants.indexed_uids(variants.path_for(lib.path)),
            uid in music_index.indexed_uids(music_index.path_for(lib.path)))


@changes(VARIANTS)
def drop_from_indexes(lib, uids: list[str], *, music: bool = True, fingerprints: bool = True,
                      mix: bool = True) -> None:
    """Tracks out of Search by Music's index, the Variants' (their fingerprints and relations) and the mix
    candidates'."""
    for on, path, open_writer, drop in ((music, music_index.path_for(lib.path), music_index.open_writer,
                                         music_index.drop),
                                        (fingerprints, variants.path_for(lib.path), variants.open_writer,
                                         variants.drop),
                                        (mix, mixability.path_for(lib.path), mixability.open_writer,
                                         mixability.drop)):
        if not on or not uids or not os.path.isfile(path):
            continue
        con = open_writer(path)
        try:
            with con:
                for uid in uids:
                    drop(con, uid)
        finally:
            con.close()


def index_status(lib) -> tuple[int, int]:
    """(library tracks indexed for Search by Music, the variants and the mix candidates, library tracks)."""
    uids = {row.uid for row in lib.db.list_all()}
    indexed = music_index.indexed_uids(music_index.path_for(lib.path)) & \
        variants.indexed_uids(variants.path_for(lib.path)) & set(mixability.feature_beats(mixability.path_for(lib.path)))
    return len(uids & indexed), len(uids)


@dataclass(frozen=True)
class IndexPlan:
    """What an Index Now run does."""
    todo: tuple          # IndexTrack
    drop: tuple          # uids gone from the library: out of Search by Music's index
    drop_variants: tuple  # and out of the variants'
    drop_mix: tuple = ()  # and out of the mix candidates'


def index_plan(lib) -> IndexPlan:
    """The library tracks to index (their audio there for Search by Music and the mix candidates' beats' features,
    those given the beats they are read on; their stored analyses for the variants) and the ones gone to drop."""
    search_db, variants_db, mix_db = (music_index.path_for(lib.path), variants.path_for(lib.path),
                                      mixability.path_for(lib.path))
    rows = lib.db.list_all(order_by="added_ts ASC")
    have, have_variants = music_index.indexed_uids(search_db), variants.indexed_uids(variants_db)
    have_mix = mixability.feature_beats(mix_db)
    uids = {row.uid for row in rows}
    todo = []
    for row in rows:
        audio = bool(row.path) and os.path.isfile(row.path)
        analysed = os.path.isfile(lib.features.path(row.uid))
        music = row.uid not in have and audio
        fp = row.uid not in have_variants and analysed
        beats = None
        if audio and analysed:
            try:
                stored_beats = analysis_jobs.mix_source(lib, row.uid).beats
            except Exception as exc:
                print(f"[Mix candidates] a stored analysis not read: {exc!r}")
                stored_beats = None
            if stored_beats is not None and stored_beats.size > 2 and not (
                    row.uid in have_mix and mixability.same_beats(have_mix[row.uid], stored_beats)):
                beats = stored_beats
        if music or fp or beats is not None:
            todo.append(IndexTrack(row.uid, row.path, int(row.file_size or 0), float(row.file_mtime or 0.0),
                                   music, fp, beats))
    return IndexPlan(tuple(todo), tuple(sorted(have - uids)),
                     tuple(sorted(variants.indexed_uids(variants_db, any_method=True) - uids)),
                     tuple(sorted(mixability.stored_uids(mix_db) - uids)))


@changes(VARIANTS)
def drop_gone(lib, plan: IndexPlan) -> None:
    """The tracks an Index Now plan found gone from the library dropped from each index."""
    drop_from_indexes(lib, list(plan.drop), fingerprints=False, mix=False)
    drop_from_indexes(lib, list(plan.drop_variants), music=False, mix=False)
    drop_from_indexes(lib, list(plan.drop_mix), music=False, fingerprints=False)


def _put_mix(lib, uid: str, beat_times, features: mix_fp.BeatFeatures, stored: StoredAnalysis) -> None:
    """A track's beats' features and the bars made of them with its stored analysis, into the mix candidates'
    index."""
    if stored.wave is None:
        raise ValueError(f"no stored overview for the mix candidates' bars: {uid}")
    bars = mix_fp.bars(stored.beats, stored.tempo_segments, stored.wave, stored.duration_sec, features)
    con = mixability.open_writer(mixability.path_for(lib.path))
    try:
        with con:
            mixability.put(con, uid, beat_times, features, bars)
    finally:
        con.close()


@changes()
def store_mix_features(lib, uid: str, beat_times, features: mix_fp.BeatFeatures) -> None:
    """A track's beats' features (read from its audio in a worker process, on ``beat_times``) and its bars into the
    mix candidates' index. ValueError when its stored beats are no longer those (edited meanwhile)."""
    stored = analysis_jobs.mix_source(lib, uid)
    if not mixability.same_beats(beat_times, stored.beats):
        raise ValueError(f"the beats changed while their features were read: {uid}")
    _put_mix(lib, uid, beat_times, features, stored)


def mix_bars_again(lib, uid: str, stored: StoredAnalysis) -> bool:
    """A track's bars made again from its stored features and analysis (its downbeats or overview as they are now),
    when its beats are still those the features were read on. Whether they were."""
    con = mixability.open_writer(mixability.path_for(lib.path))
    try:
        kept = mixability.features_of(con, uid)
        if kept is None or not mixability.same_beats(kept[0], stored.beats) or stored.wave is None:
            return False
        bars = mix_fp.bars(stored.beats, stored.tempo_segments, stored.wave, stored.duration_sec, kept[1])
        with con:
            mixability.put_bars(con, uid, bars)
        return True
    finally:
        con.close()


@changes()
def stale_mix_bars_again(lib) -> int:
    """Index Now done: the bars of an earlier method made again from the features kept (no audio). How many."""
    done = 0
    for uid in mixability.stale_bars(mixability.path_for(lib.path)):
        try:
            done += mix_bars_again(lib, uid, analysis_jobs.mix_source(lib, uid))
        except Exception as exc:
            print(f"[Mix candidates] bars not made again: {exc}")
    return done


def mix_candidates(lib, uid: str, options: mix_search.MixOptions) -> list[mix_search.Candidate]:
    """The loaded track's mix candidates (core.search.mixability): every library track but it, its versions (the
    Variants) and the instrumentals (by their titles), by ``options``; best first."""
    rows = lib.db.list_all()
    marks = lib.db.mix_marks()

    def track_marks(row: TrackRow) -> mix_search.TrackMarks:
        phrases, tempo, keys = marks.get(row.uid, ([], [], []))
        return mix_search.TrackMarks(row.uid, row.title or "", tuple(mix_fp.Unit(*p) for p in phrases),
                                     tuple(tempo), tuple(keys))

    by_uid = {row.uid: row for row in rows}
    if uid not in by_uid:
        return []
    uids = set(by_uid)
    pairs = [p for p in variants.pairs(variants.path_for(lib.path)) if p.a in uids and p.b in uids]
    excluded = {u for g in variant_groups(pairs, uids) if uid in g for u in g}
    return mix_search.candidates(track_marks(by_uid[uid]), [track_marks(r) for r in rows],
                                 mixability.all_bars(mixability.path_for(lib.path)), excluded, options)


@changes()
def store_music_entry(lib, track: IndexTrack, entry) -> None:
    """A track's Search by Music entry (made from its audio in a worker process) into the index."""
    con = music_index.open_writer(music_index.path_for(lib.path))
    try:
        with con:
            music_index.put(con, track.uid, track.size, track.mtime, entry)
    finally:
        con.close()


@changes(VARIANTS)
def fingerprint(lib, uid: str) -> int:
    """A track's variants fingerprint from its stored analysis (its beats and overview), queried against the
    library's. The versions found."""
    stored = analysis_jobs.fingerprint_source(lib, uid)
    if stored.wave is None:
        raise ValueError(f"no stored overview to fingerprint: {uid}")
    con = variants.open_writer(variants.path_for(lib.path))
    try:
        return variants.put(con, uid, variant_fp.track_fingerprints(stored.beats, stored.wave, stored.duration_sec))
    finally:
        con.close()


@changes(VARIANTS)
def prune_fingerprints(lib) -> None:
    """Index Now done: the fingerprints of earlier methods the current one replaced removed."""
    con = variants.open_writer(variants.path_for(lib.path))
    try:
        variants.prune_old_methods(con)
    finally:
        con.close()


# ---- the stores beside library.db checked against its tracks (at the start) ----

ORPHANED_ANALYSES = package.ORPHANED     # where a stored analysis of no library track is moved (kept, never read)


@dataclass(frozen=True)
class StoreCheck:
    """What the start's check of the library's stores found left over (of tracks not in library.db) and dropped."""
    rows: int = 0            # library.db rows (segments, readings, tag sources, tie-ups, playlist entries)
    music: int = 0           # Search by Music's index
    variants: int = 0        # the Variants' index
    thumbnails: int = 0
    analyses: int = 0        # stored analyses moved into ORPHANED_ANALYSES
    skipped: bool = False    # library.db holds no track but the stores do: nothing touched
    mix: int = 0             # the mix candidates' index

    def total(self) -> int:
        return self.rows + self.music + self.variants + self.mix + self.thumbnails + self.analyses


@changes(PLAYLISTS, READINGS, TIEUPS, VARIANTS)
def check_stores(lib) -> StoreCheck:
    """Every store beside library.db made to agree with its tracks: what belongs to no track (a removal cut short, a
    file in use then, an earlier version's) dropped, a stored analysis set aside instead (it may hold the user's
    edits). A library.db without tracks beside stores that have some is taken for a library.db lost, not leftovers:
    nothing touched. Said on the console."""
    uids = {row.uid for row in lib.db.list_all()}
    music_db, variants_db = music_index.path_for(lib.path), variants.path_for(lib.path)
    music = music_index.stored_uids(music_db) - uids
    fingerprinted = (variants.indexed_uids(variants_db, any_method=True)
                     | {u for p in variants.pairs(variants_db) for u in (p.a, p.b)}) - uids
    mixed = mixability.stored_uids(mixability.path_for(lib.path)) - uids
    thumbs = thumbnails.thumbnail_uids(lib.path) - uids
    analyses = lib.features.uids() - uids
    if not uids and (music or fingerprinted or mixed or thumbs or analyses):
        print(f"[Library] library.db holds no track, but {len(analyses)} stored analyses and "
              f"{len(music | fingerprinted | mixed | thumbs)} index entries or thumbnails do: left as they are")
        return StoreCheck(skipped=True)
    rows = lib.db.drop_orphan_rows()
    drop_from_indexes(lib, sorted(music), fingerprints=False, mix=False)
    drop_from_indexes(lib, sorted(fingerprinted), music=False, mix=False)
    drop_from_indexes(lib, sorted(mixed), music=False, fingerprints=False)
    dropped = 0
    for uid in thumbs:
        try:        # a file another program holds (Explorer, an antivirus): left for the next start
            thumbnails.delete_thumbnail(lib.path, uid)
            dropped += 1
        except OSError as exc:
            print(f"[Library] a thumbnail of no track left in place: {exc}")
    moved = 0
    for uid in sorted(analyses):
        try:
            lib.features.set_aside(uid, ORPHANED_ANALYSES)
            moved += 1
        except OSError as exc:
            print(f"[Library] a stored analysis of no track left in place: {exc}")
    found = StoreCheck(rows, len(music), len(fingerprinted), dropped, moved, mix=len(mixed))
    if found.total():
        print(f"[Library] left over by tracks no longer in the library: {rows} rows, {len(music)} Search by Music "
              f"entries, {len(fingerprinted)} Variants fingerprints, {len(mixed)} mix candidates' entries, "
              f"{dropped} thumbnails dropped; {moved} stored analyses moved to "
              f"{os.path.join(lib.path, ORPHANED_ANALYSES)}")
    else:
        print(f"[Library] the stores agree with library.db ({len(uids)} tracks)")
    return found


def open_music_search(lib):
    """Search by Music's view of its index, (LandmarkIndex, MelodyIndex) or None: the landmarks on a read-only
    connection lent to the search's thread (it reads them every few seconds: on the owner's thread they would hold
    its other jobs)."""
    return music_index.open_search(music_index.path_for(lib.path))


# ---- the library package: a backup, a move to another device (core.library.package) ----

def export_package(lib, plan: package.ExportPlan, progress=None, cancelled=None) -> package.ExportResult:
    """The library packed into ``plan.dest`` with what the plan adds (the settings, the user's files, the caches).
    Asked as a read: no other job runs while it is written, so library.db and the stored analyses are of one moment.
    ``progress(share)`` and ``cancelled()`` are called on the owner's thread."""
    return package.export(lib, plan, progress, cancelled)


# ---- the thumbnails (the stored analyses' jobs are core.analysis_library.jobs') ----

def thumbnail(lib, uid: str, width_px: int) -> Optional[QtGui.QImage]:
    """A track's thumbnail scaled to ``width_px``, None when it has none."""
    image = thumbnails.load_thumbnail(lib.path, uid)
    return image.scaledToWidth(max(1, int(width_px)), QtCore.Qt.SmoothTransformation) if image is not None else None


# ---- an analysis's results ----

def _kept_metadata(properties: dict, track: Optional[TrackRow]) -> dict:
    """A track already in the library keeps the user's metadata (title, artist, album, rating, comment) over what the
    file's tags say: a reanalysis, and an analysis of a track whose stored one did not read."""
    if track is None:
        return properties
    merged = dict(properties)
    for key in ("album", "artist", "rating", "title", "comment"):
        merged[key] = getattr(track, key, merged.get(key))
    return merged


def store_analysis(lib, payload: dict) -> dict:
    """An analysis's results stored (the analysis process never opens the library: its last item carries ``store``):
    the track's row (one in the library keeps the user's metadata; a new track given its uid), its tag sources, its
    rows made from the marks (``store["index"]``, segments.track_index: written as the editor's save and the sync
    write them) and its stored analysis (``arrays``, the NPZ), in one transaction (the NPZ last); then its thumbnail,
    Search by Music's entry, the Variants' fingerprint (queried against the library's), and a new track's versions'
    tie-ups. Returns the payload with its properties' uid, ``analysis`` (the StoredAnalysis of the arrays stored,
    made from them in memory; None when they do not read), ``new_track`` and ``tieups`` (taken from its
    versions)."""
    payload = dict(payload)
    stored = payload.pop("store")
    arrays = payload.pop("arrays")
    db = lib.db
    properties = dict(payload["properties"])
    known = db.get(properties["path"])
    properties = _kept_metadata(properties, known)
    if known is not None and known.uid:
        properties["uid"] = known.uid
    with db.tx():
        uid = db.upsert_meta(properties)
        properties["uid"] = uid
        if properties.get("tag_sources"):
            db.set_tag_sources(uid, properties["tag_sources"])
        db.write_track_index(uid, stored["index"])
        lib.features.save(uid, dict(arrays))
    # the analysis as a library track's is read (StoredAnalysis), from the arrays just stored: no second read of the
    # file; one that does not read leaves the deck without it, never the rest of the store undone
    try:
        payload["analysis"] = StoredAnalysis.from_arrays(arrays)
    except Exception as exc:
        payload["analysis"] = None
        print(f"[Library] the stored analysis could not be read for the deck: {exc}")
    cover = stored.get("cover")
    thumbnails.save_thumbnail(lib.path, uid, QtGui.QImage.fromData(*cover) if cover is not None else None)
    if stored["music_entry"] is not None:
        try:
            store_music_entry(lib, IndexTrack(uid, properties["path"], int(properties.get("file_size") or 0),
                                              float(properties.get("file_mtime") or 0.0), True, False),
                              stored["music_entry"])
        except Exception as exc:
            print(f"[Music search] index not written: {exc}")
    # the variants' fingerprints (the analysis's: analyzer.search.variants), queried against the library's; a failure
    # leaves the track out of the variants, never the analysis undone
    if stored.get("variants"):
        try:
            con = variants.open_writer(variants.path_for(lib.path))
            try:
                found = variants.put(con, uid, stored["variants"])
            finally:
                con.close()
            if found:
                print(f"[Variants] {found} found")
        except Exception as exc:
            print(f"[Variants] not indexed: {exc}")
    # the mix candidates' beats' features (the analysis's) and their bars; a failure leaves the track out of the
    # candidates until Index Now, never the analysis undone
    if stored.get("mix") is not None:
        try:
            analysis = payload["analysis"] or StoredAnalysis.from_arrays(arrays)
            _put_mix(lib, uid, analysis.beats, stored["mix"], analysis)
        except Exception as exc:
            print(f"[Mix candidates] not indexed: {exc}")
    # a new track (not in the library before this analysis) takes its versions' tie-ups; a failure leaves it without,
    # never the analysis undone
    payload["new_track"] = known is None
    if known is None:
        try:
            known = {row.uid for row in db.list_all()}
            added = tieups.match_new_track(db.conn, uid, [p for p in variants.pairs(variants.path_for(lib.path))
                                                          if p.a in known and p.b in known])
            db.conn.commit()
            payload["tieups"] = added      # none: the GUI may look on the web (core.workflows.tieup_web)
            if added:
                print(f"[Tie-ups] {added} taken from its versions")
        except Exception as exc:
            db.conn.rollback()
            print(f"[Tie-ups] not taken from its versions: {exc}")
    payload["properties"] = properties
    return payload
