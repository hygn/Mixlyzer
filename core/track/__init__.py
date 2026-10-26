"""The deck's loaded track: one owner of what it is and of its marks as edited. Where to look:

    model.py      DataModel, the owner: the track's stored analysis, properties, album art and decoded audio; its
                  marks (beatgrid and tempo rows, keys, phrases, JumpCUEs, CUE points) changed only through its set_*
                  methods, each telling its own signal; undo / redo and the unsaved flag
    marks.py      the marks' editor form, made from the stored analysis (pure functions)
    history.py    the edit history's snapshots

The editor (ui.edit_panel) shows the marks and asks the model to change them; a segment reanalysis's result goes
to the model the same way; the views read the model and follow its signals. Saving reads DataModel.stored_edits (a TrackEdits).
"""
