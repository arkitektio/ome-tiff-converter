"""The app's logo, and one logo per action

The app's logo is a LOGO blok (an ``<AgentLogo>`` root): an SVG in colours of its own,
which a client shows beside the app's name. Each action's logo is an ACTION_LOGO blok
(an ``<ActionLogo>`` root): the same SVG shapes, but painted only with the viewer's palette (``ink``, ``muted``, ``brand``,
``brand-strong``, ``brand-soft``, ``surface``), so the actions of every app look like
one set and follow the viewer's brand and dark mode. A shape that ``animate``s plays
while the logo is hovered, and while the viewer's run of the action is going.

Every logo is drawn on a 24 x 24 grid to read at 16 px. Blok names are the
organization's, so each carries the app's name.
"""

from arkitekt import App

APP = "ome_tiff_converter"

APP_LOGO = f"{APP}_logo"

# An OME-TIFF, its corner folded and its OME-XML header in brackets, beside a stack of
# planes in three fluorescence channels: one file in, a dataset per Image out.
APP_LOGO_SOURCE = """
<AgentLogo viewBox="0 0 64 64" title="ome-tiff-converter">
  <Defs>
    <LinearGradient id="ground" x1="0" y1="0" x2="1" y2="1">
      <Stop offset="0" stopColor="ink" />
      <Stop offset="1" stopColor="ink" />
    </LinearGradient>
  </Defs>
  <Rect x="0" y="0" width="64" height="64" rx="14" fill="url(#ground)" />
  <Path d="M9 11a3 3 0 0 1 3-3h12l7 7v17a3 3 0 0 1-3 3H12a3 3 0 0 1-3-3Z" fill="surface" />
  <Path d="M24 8v5a2 2 0 0 0 2 2h5" fill="muted" />
  <G fill="none" stroke="muted" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <Path d="M15.5 18.5 12.5 22l3 3.5" />
    <Path d="M24.5 18.5 27.5 22l-3 3.5" />
  </G>
  <Rect x="17.5" y="21" width="5" height="2.2" rx="1.1" fill="muted" />
  <Path d="M35 12.5h9a5 5 0 0 1 5 5V21" fill="none" stroke="muted" strokeWidth="2.4" strokeLinecap="round" />
  <Path d="M45.5 18.5 49 22l3.5-3.5" fill="none" stroke="muted" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
  <G stroke="ink" strokeWidth="1.6">
    <Rect x="32" y="28" width="22" height="22" rx="3" fill="spectrum-red" />
    <Rect x="27" y="33" width="22" height="22" rx="3" fill="spectrum-green" />
    <Rect x="22" y="38" width="22" height="22" rx="3" fill="spectrum-cyan" />
  </G>
  <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
</AgentLogo>
"""

ACTION_LOGOS: dict[str, str] = {
    # A file, its corner folded, and the planes of its images lifting out of it.
    "convert_ome_tiff_file_to_datasets": """
<ActionLogo viewBox="0 0 24 24" title="Convert OME-TIFF file to Datasets">
  <Path d="M3 4a1.5 1.5 0 0 1 1.5-1.5H9l3 3V12a1.5 1.5 0 0 1-1.5 1.5h-6A1.5 1.5 0 0 1 3 12Z" fill="surface" stroke="ink" strokeWidth="1.5" strokeLinejoin="round" />
  <Path d="M9 2.5V5.5H12" fill="none" stroke="ink" strokeWidth="1.5" strokeLinejoin="round" />
  <Path d="M14 4h3a2 2 0 0 1 2 2v1.5" fill="none" stroke="brand-strong" strokeWidth="1.6" strokeLinecap="round" animate="draw" />
  <Path d="M17.25 6 19 7.75 20.75 6" fill="none" stroke="brand-strong" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
  <G stroke="ink" strokeWidth="1.4" strokeLinejoin="round">
    <Rect x="13.5" y="10.5" width="8" height="8" rx="1.2" fill="brand-soft" animate="float" animateDelay="0.3" />
    <Rect x="11.5" y="12.5" width="8" height="8" rx="1.2" fill="brand" animate="float" animateDelay="0.15" />
    <Rect x="9.5" y="14.5" width="8" height="8" rx="1.2" fill="brand-strong" animate="float" />
  </G>
</ActionLogo>
""",
}


def logo_of(action: str) -> str:
    """The name of the action logo blok that draws ``action`` (its function's name)."""
    if action not in ACTION_LOGOS:
        raise KeyError(f"No logo is drawn for the action {action!r}")
    return f"{APP}_{action}"


def declare_logos(app: App) -> None:
    """Declares the app's logo, as its default, and every action logo."""
    app.blok(
        APP_LOGO,
        APP_LOGO_SOURCE,
        description="An OME-TIFF beside a stack of planes in three channels.",
        default=True,
    )
    for action, source in ACTION_LOGOS.items():
        app.blok(logo_of(action), source)
