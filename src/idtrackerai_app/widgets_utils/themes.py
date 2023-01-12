from PyQt6.QtGui import QColor, QPalette

ColorRole = QPalette.ColorRole
# style = """
# QCheckBox::indicator {
#     width: 11px;
#     height: 11px;
#     background-color: #4C4D5A;
#     border-radius: 3px;
#     border-style: solid;
#     border-width: 2px;
#     border-color: #8A94B7 #8A94B7 #8A94C7 #8A94C7;
# }
# QCheckBox::indicator:checked {
#     background-color: #8AB4F7;
# }
# QCheckBox:checked, QCheckBox::indicator:checked {
#     border-color: #202020 #202020 #202020 #202020;
# }
# """


def apply_style(app, style="custom"):

    if style == "custom":
        app.setPalette(custom_palette())
    elif style == "dark":
        app.setPalette(dark_palette())
    elif style == "light":
        app.setPalette(light_palette())
    else:
        raise ValueError(f"{style} not in ('custom', 'dark, 'light')")


def custom_palette() -> QPalette:
    Disabled = QPalette.ColorGroup.Disabled
    palette = QPalette()
    background = "#202124"
    mid_background = "#2C2D2F"
    light_bkg = "#3F4042"
    blue = "#8AB4F7"
    blue = "#AAD4FF"
    almost_white = "#FDFDFD"
    placeholder_color = "#B0B0B0"
    red = "#FF0000"
    palette.setColor(ColorRole.Window, QColor(background))
    palette.setColor(ColorRole.WindowText, QColor(almost_white))
    palette.setColor(ColorRole.Base, QColor(light_bkg))
    palette.setColor(ColorRole.AlternateBase, QColor(mid_background))
    palette.setColor(ColorRole.Text, QColor(almost_white))
    palette.setColor(ColorRole.Button, QColor(mid_background))
    palette.setColor(ColorRole.ButtonText, QColor(blue))
    palette.setColor(ColorRole.Highlight, QColor(blue))
    palette.setColor(ColorRole.HighlightedText, QColor(background))
    palette.setColor(ColorRole.PlaceholderText, QColor(placeholder_color))
    palette.setColor(
        Disabled, ColorRole.WindowText, palette.windowText().color().darker()
    )
    palette.setColor(Disabled, ColorRole.Button, palette.button().color().darker())
    palette.setColor(
        Disabled, ColorRole.ButtonText, palette.buttonText().color().darker()
    )
    palette.setColor(Disabled, ColorRole.Base, palette.base().color().darker())
    palette.setColor(Disabled, ColorRole.Text, palette.text().color().darker())
    palette.setColor(ColorRole.BrightText, QColor(red))
    palette.setColor(ColorRole.Light, QColor(red))
    palette.setColor(ColorRole.Midlight, QColor(red))
    palette.setColor(ColorRole.Dark, QColor(red))
    palette.setColor(ColorRole.Shadow, QColor(red))
    palette.setColor(ColorRole.Mid, QColor(red))
    palette.setColor(ColorRole.Link, QColor(red))
    palette.setColor(ColorRole.LinkVisited, QColor(red))
    palette.setColor(ColorRole.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(ColorRole.ToolTipText, QColor("#000000"))
    palette.setColor(ColorRole.NoRole, QColor(red))
    return palette


def dark_palette() -> QPalette:
    Disabled = QPalette.ColorGroup.Disabled
    palette = QPalette()
    palette.setColor(ColorRole.WindowText, QColor("#f0f0f0"))
    palette.setColor(ColorRole.Button, QColor("#323232"))
    palette.setColor(ColorRole.Light, QColor("#4b4b4b"))
    palette.setColor(ColorRole.Midlight, QColor("#2a2a2a"))
    palette.setColor(ColorRole.Dark, QColor("#212121"))
    palette.setColor(ColorRole.Mid, QColor("#262626"))
    palette.setColor(ColorRole.Text, QColor("#f0f0f0"))
    palette.setColor(ColorRole.BrightText, QColor("#4b4b4b"))
    palette.setColor(ColorRole.ButtonText, QColor("#f0f0f0"))
    palette.setColor(ColorRole.Base, QColor("#242424"))
    palette.setColor(ColorRole.Window, QColor("#323232"))
    palette.setColor(ColorRole.Shadow, QColor("#191919"))
    palette.setColor(ColorRole.Highlight, QColor("#308cc6"))
    palette.setColor(ColorRole.HighlightedText, QColor("#303030"))
    palette.setColor(ColorRole.Link, QColor("#308cc6"))
    palette.setColor(ColorRole.LinkVisited, QColor("#ff00ff"))
    palette.setColor(ColorRole.AlternateBase, QColor("#2b2b2b"))
    palette.setColor(ColorRole.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(ColorRole.ToolTipText, QColor("#000000"))
    palette.setColor(ColorRole.PlaceholderText, QColor("#f0f0f0"))
    palette.setColor(ColorRole.NoRole, QColor("#000000"))
    palette.setColor(Disabled, ColorRole.WindowText, QColor("#828282"))
    palette.setColor(Disabled, ColorRole.Button, QColor("#323232"))
    palette.setColor(Disabled, ColorRole.Light, QColor("#4b4b4b"))
    palette.setColor(Disabled, ColorRole.Midlight, QColor("#2a2a2a"))
    palette.setColor(Disabled, ColorRole.Dark, QColor("#bebebe"))
    palette.setColor(Disabled, ColorRole.Mid, QColor("#262626"))
    palette.setColor(Disabled, ColorRole.Text, QColor("#828282"))
    palette.setColor(Disabled, ColorRole.BrightText, QColor("#4b4b4b"))
    palette.setColor(Disabled, ColorRole.ButtonText, QColor("#828282"))
    palette.setColor(Disabled, ColorRole.Base, QColor("#323232"))
    palette.setColor(Disabled, ColorRole.Window, QColor("#323232"))
    palette.setColor(Disabled, ColorRole.Shadow, QColor("#252525"))
    palette.setColor(Disabled, ColorRole.Highlight, QColor("#919191"))
    palette.setColor(Disabled, ColorRole.HighlightedText, QColor("#f0f0f0"))
    palette.setColor(Disabled, ColorRole.Link, QColor("#308cc6"))
    palette.setColor(Disabled, ColorRole.LinkVisited, QColor("#ff00ff"))
    palette.setColor(Disabled, ColorRole.AlternateBase, QColor("#2b2b2b"))
    palette.setColor(Disabled, ColorRole.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(Disabled, ColorRole.ToolTipText, QColor("#000000"))
    palette.setColor(Disabled, ColorRole.PlaceholderText, QColor("#f0f0f0"))
    palette.setColor(Disabled, ColorRole.NoRole, QColor("#000000"))
    return palette


def light_palette() -> QPalette:
    Disabled = QPalette.ColorGroup.Disabled
    palette = QPalette()
    palette.setColor(ColorRole.WindowText, QColor("#000000"))
    palette.setColor(ColorRole.Button, QColor("#efefef"))
    palette.setColor(ColorRole.Light, QColor("#ffffff"))
    palette.setColor(ColorRole.Midlight, QColor("#cacaca"))
    palette.setColor(ColorRole.Dark, QColor("#9f9f9f"))
    palette.setColor(ColorRole.Mid, QColor("#b8b8b8"))
    palette.setColor(ColorRole.Text, QColor("#000000"))
    palette.setColor(ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(ColorRole.ButtonText, QColor("#000000"))
    palette.setColor(ColorRole.Base, QColor("#ffffff"))
    palette.setColor(ColorRole.Window, QColor("#efefef"))
    palette.setColor(ColorRole.Shadow, QColor("#767676"))
    palette.setColor(ColorRole.Highlight, QColor("#308cc6"))
    palette.setColor(ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(ColorRole.Link, QColor("#0000ff"))
    palette.setColor(ColorRole.LinkVisited, QColor("#ff00ff"))
    palette.setColor(ColorRole.AlternateBase, QColor("#f7f7f7"))
    palette.setColor(ColorRole.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(ColorRole.ToolTipText, QColor("#000000"))
    palette.setColor(ColorRole.PlaceholderText, QColor("#000000"))
    palette.setColor(ColorRole.NoRole, QColor("#000000"))
    palette.setColor(Disabled, ColorRole.WindowText, QColor("#bebebe"))
    palette.setColor(Disabled, ColorRole.Button, QColor("#efefef"))
    palette.setColor(Disabled, ColorRole.Light, QColor("#ffffff"))
    palette.setColor(Disabled, ColorRole.Midlight, QColor("#cacaca"))
    palette.setColor(Disabled, ColorRole.Dark, QColor("#bebebe"))
    palette.setColor(Disabled, ColorRole.Mid, QColor("#b8b8b8"))
    palette.setColor(Disabled, ColorRole.Text, QColor("#bebebe"))
    palette.setColor(Disabled, ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(Disabled, ColorRole.ButtonText, QColor("#bebebe"))
    palette.setColor(Disabled, ColorRole.Base, QColor("#efefef"))
    palette.setColor(Disabled, ColorRole.Window, QColor("#efefef"))
    palette.setColor(Disabled, ColorRole.Shadow, QColor("#b1b1b1"))
    palette.setColor(Disabled, ColorRole.Highlight, QColor("#919191"))
    palette.setColor(Disabled, ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(Disabled, ColorRole.Link, QColor("#0000ff"))
    palette.setColor(Disabled, ColorRole.LinkVisited, QColor("#ff00ff"))
    palette.setColor(Disabled, ColorRole.AlternateBase, QColor("#f7f7f7"))
    palette.setColor(Disabled, ColorRole.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(Disabled, ColorRole.ToolTipText, QColor("#000000"))
    palette.setColor(Disabled, ColorRole.PlaceholderText, QColor("#000000"))
    palette.setColor(Disabled, ColorRole.NoRole, QColor("#000000"))
    return palette
