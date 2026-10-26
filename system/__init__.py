"""The lowest layer: what every other package may use and which imports nothing of the app.

    resource_paths  the bundled files and the app's folder
    os_support      the operating system, detected once, and what differs by it
    i18n            the UI's language and its catalogs (tr)
    model_files     a learned model's file: the user's, else the shipped default
    atomic_io       files replaced whole or not at all
    process_worker  a spawned worker process's Qt lifecycle (its messages, its end)

The layers, each importing only those before it (tests/test_layers.py): system -> analyzer, audioengine -> core ->
views, ui -> app.
"""
