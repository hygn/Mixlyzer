"""The library's widget (ui.library.LibraryWidget): the track table, the sources panel, the search bar, Search by
Music, the playlists, the track panel and the tracks' actions, a module each.

    widget       LibraryWidget, built from the parts below (mixins, as ui/edit_panel's)
    common       what they share: a track as shown, the roles and timings
    table        the track table (painting, drag and drop)
    sources      the sources panel: the library, Variants, playlists, folders
    search       the search bar: text search, filters, the structured searches
    music        Search by Music in the search bar
    playlists    the playlists
    track_panel  the track panel's tabs
    actions      the tracks' actions and the selection they act on
"""
from ui.library.widget import LibraryWidget

__all__ = ["LibraryWidget"]
