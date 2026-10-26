# Third party components and lesson content

Project credits: Jeff Barrientos, Hannah Dimaculangan, and Trixie Mendoza.

Senyas installs Python and third-party packages from their official distribution services. Their copyright and license notices remain in the installed runtime and each package's metadata. Project credits do not claim authorship of those libraries.

The bundled lesson demonstrations are derived from the supplied FSL105 collection. `assets/lessons/manifest.json` preserves the original relative clip mapping and output checksums. Audio tracks were removed; numeric class IDs were preserved. No new license is asserted for the original dataset or its performances.

The Python runtime is pinned for compatibility with the existing MediaPipe and scikit-learn models. Python distribution information is available at https://www.python.org/downloads/release/python-3119/ and https://docs.python.org/3.11/license.html. Package versions and download checksums are recorded in `requirements-windows.lock`.
