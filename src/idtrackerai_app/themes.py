from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QStyleFactory
import logging


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
    Disabled = QPalette.Disabled
    palette = QPalette()
    background = "#202124"
    mid_background = "#2C2D2F"
    light_bkg = "#3F4042"
    blue = "#8AB4F7"
    blue = "#AAD4FF"
    almost_white = "#FDFDFD"
    placeholder_color = "#B0B0B0"
    red = "#FF0000"
    palette.setColor(QPalette.Window, QColor(background))
    palette.setColor(QPalette.WindowText, QColor(almost_white))
    palette.setColor(QPalette.Base, QColor(light_bkg))
    palette.setColor(QPalette.AlternateBase, QColor(mid_background))
    palette.setColor(QPalette.Text, QColor(almost_white))
    palette.setColor(QPalette.Button, QColor(mid_background))
    palette.setColor(QPalette.ButtonText, QColor(blue))
    palette.setColor(QPalette.Highlight, QColor(blue))
    palette.setColor(QPalette.HighlightedText, QColor(background))
    palette.setColor(QPalette.PlaceholderText, QColor(placeholder_color))
    palette.setColor(
        Disabled, QPalette.WindowText, palette.windowText().color().darker()
    )
    palette.setColor(
        Disabled, QPalette.Button, palette.button().color().darker()
    )
    palette.setColor(
        Disabled, QPalette.ButtonText, palette.buttonText().color().darker()
    )
    palette.setColor(Disabled, QPalette.Base, palette.base().color().darker())
    palette.setColor(Disabled, QPalette.Text, palette.text().color().darker())
    palette.setColor(QPalette.BrightText, QColor(red))
    palette.setColor(QPalette.Light, QColor(red))
    palette.setColor(QPalette.Midlight, QColor(red))
    palette.setColor(QPalette.Dark, QColor(red))
    palette.setColor(QPalette.Shadow, QColor(red))
    palette.setColor(QPalette.Mid, QColor(red))
    palette.setColor(QPalette.Link, QColor(red))
    palette.setColor(QPalette.LinkVisited, QColor(red))
    palette.setColor(QPalette.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(QPalette.ToolTipText, QColor("#000000"))
    palette.setColor(QPalette.NoRole, QColor(red))
    return palette


def dark_palette() -> QPalette:
    Disabled = QPalette.Disabled
    palette = QPalette()
    palette.setColor(QPalette.WindowText, QColor("#f0f0f0"))
    palette.setColor(QPalette.Button, QColor("#323232"))
    palette.setColor(QPalette.Light, QColor("#4b4b4b"))
    palette.setColor(QPalette.Midlight, QColor("#2a2a2a"))
    palette.setColor(QPalette.Dark, QColor("#212121"))
    palette.setColor(QPalette.Mid, QColor("#262626"))
    palette.setColor(QPalette.Text, QColor("#f0f0f0"))
    palette.setColor(QPalette.BrightText, QColor("#4b4b4b"))
    palette.setColor(QPalette.ButtonText, QColor("#f0f0f0"))
    palette.setColor(QPalette.Base, QColor("#242424"))
    palette.setColor(QPalette.Window, QColor("#323232"))
    palette.setColor(QPalette.Shadow, QColor("#191919"))
    palette.setColor(QPalette.Highlight, QColor("#308cc6"))
    palette.setColor(QPalette.HighlightedText, QColor("#303030"))
    palette.setColor(QPalette.Link, QColor("#308cc6"))
    palette.setColor(QPalette.LinkVisited, QColor("#ff00ff"))
    palette.setColor(QPalette.AlternateBase, QColor("#2b2b2b"))
    palette.setColor(QPalette.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(QPalette.ToolTipText, QColor("#000000"))
    palette.setColor(QPalette.PlaceholderText, QColor("#f0f0f0"))
    palette.setColor(QPalette.NoRole, QColor("#000000"))
    palette.setColor(Disabled, QPalette.WindowText, QColor("#828282"))
    palette.setColor(Disabled, QPalette.Button, QColor("#323232"))
    palette.setColor(Disabled, QPalette.Light, QColor("#4b4b4b"))
    palette.setColor(Disabled, QPalette.Midlight, QColor("#2a2a2a"))
    palette.setColor(Disabled, QPalette.Dark, QColor("#bebebe"))
    palette.setColor(Disabled, QPalette.Mid, QColor("#262626"))
    palette.setColor(Disabled, QPalette.Text, QColor("#828282"))
    palette.setColor(Disabled, QPalette.BrightText, QColor("#4b4b4b"))
    palette.setColor(Disabled, QPalette.ButtonText, QColor("#828282"))
    palette.setColor(Disabled, QPalette.Base, QColor("#323232"))
    palette.setColor(Disabled, QPalette.Window, QColor("#323232"))
    palette.setColor(Disabled, QPalette.Shadow, QColor("#252525"))
    palette.setColor(Disabled, QPalette.Highlight, QColor("#919191"))
    palette.setColor(Disabled, QPalette.HighlightedText, QColor("#f0f0f0"))
    palette.setColor(Disabled, QPalette.Link, QColor("#308cc6"))
    palette.setColor(Disabled, QPalette.LinkVisited, QColor("#ff00ff"))
    palette.setColor(Disabled, QPalette.AlternateBase, QColor("#2b2b2b"))
    palette.setColor(Disabled, QPalette.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(Disabled, QPalette.ToolTipText, QColor("#000000"))
    palette.setColor(Disabled, QPalette.PlaceholderText, QColor("#f0f0f0"))
    palette.setColor(Disabled, QPalette.NoRole, QColor("#000000"))
    return palette


def light_palette() -> QPalette:
    Disabled = QPalette.Disabled
    palette = QPalette()
    palette.setColor(QPalette.WindowText, QColor("#000000"))
    palette.setColor(QPalette.Button, QColor("#efefef"))
    palette.setColor(QPalette.Light, QColor("#ffffff"))
    palette.setColor(QPalette.Midlight, QColor("#cacaca"))
    palette.setColor(QPalette.Dark, QColor("#9f9f9f"))
    palette.setColor(QPalette.Mid, QColor("#b8b8b8"))
    palette.setColor(QPalette.Text, QColor("#000000"))
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ButtonText, QColor("#000000"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.Window, QColor("#efefef"))
    palette.setColor(QPalette.Shadow, QColor("#767676"))
    palette.setColor(QPalette.Highlight, QColor("#308cc6"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Link, QColor("#0000ff"))
    palette.setColor(QPalette.LinkVisited, QColor("#ff00ff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f7f7f7"))
    palette.setColor(QPalette.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(QPalette.ToolTipText, QColor("#000000"))
    palette.setColor(QPalette.PlaceholderText, QColor("#000000"))
    palette.setColor(QPalette.NoRole, QColor("#000000"))
    palette.setColor(Disabled, QPalette.WindowText, QColor("#bebebe"))
    palette.setColor(Disabled, QPalette.Button, QColor("#efefef"))
    palette.setColor(Disabled, QPalette.Light, QColor("#ffffff"))
    palette.setColor(Disabled, QPalette.Midlight, QColor("#cacaca"))
    palette.setColor(Disabled, QPalette.Dark, QColor("#bebebe"))
    palette.setColor(Disabled, QPalette.Mid, QColor("#b8b8b8"))
    palette.setColor(Disabled, QPalette.Text, QColor("#bebebe"))
    palette.setColor(Disabled, QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(Disabled, QPalette.ButtonText, QColor("#bebebe"))
    palette.setColor(Disabled, QPalette.Base, QColor("#efefef"))
    palette.setColor(Disabled, QPalette.Window, QColor("#efefef"))
    palette.setColor(Disabled, QPalette.Shadow, QColor("#b1b1b1"))
    palette.setColor(Disabled, QPalette.Highlight, QColor("#919191"))
    palette.setColor(Disabled, QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(Disabled, QPalette.Link, QColor("#0000ff"))
    palette.setColor(Disabled, QPalette.LinkVisited, QColor("#ff00ff"))
    palette.setColor(Disabled, QPalette.AlternateBase, QColor("#f7f7f7"))
    palette.setColor(Disabled, QPalette.ToolTipBase, QColor("#ffffdc"))
    palette.setColor(Disabled, QPalette.ToolTipText, QColor("#000000"))
    palette.setColor(Disabled, QPalette.PlaceholderText, QColor("#000000"))
    palette.setColor(Disabled, QPalette.NoRole, QColor("#000000"))
    return palette
