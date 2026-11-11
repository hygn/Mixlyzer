"""Deck Monitor: another DJ program's decks shown beside the app, never driving it.

    decks.py       what is known of each deck, whatever protocol brought it (DeckState, TrackInfo)
    osc.py         the strict OSC 1.0 decoder the OSC protocols read with
    osc_basic.py   OSC Basic: each deck's track and position, its receiver
    match.py       a deck's track found in the library (its path, else its title, artist and length)
    controller.py  the receiver run while Deck Monitor is on, the decks matched, for the monitor's window

The app's deck, player and library are not touched: the window (ui.windows.deck_monitor) shows the decks."""
