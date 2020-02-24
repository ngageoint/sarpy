SEABORN_PALETTES = dict(
    deep=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    deep6=["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"],
    muted=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"],
    muted6=["#4878D0", "#6ACC64", "#D65F5F", "#956CB4", "#D5BB67", "#82C6E2"],
    pastel=["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF", "#DEBB9B", "#FAB0E4", "#CFCFCF", "#FFFEA3", "#B9F2F0"],
    pastel6=["#A1C9F4", "#8DE5A1", "#FF9F9B", "#D0BBFF", "#FFFEA3", "#B9F2F0"],
    bright=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2", "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"],
    bright6=["#023EFF", "#1AC938", "#E8000B", "#8B2BE2", "#FFC400", "#00D7FF"],
    dark=["#001C7F", "#B1400D", "#12711C", "#8C0800", "#591E71", "#592F0D", "#A23582", "#3C3C3C", "#B8850A", "#006374"],
    dark6=["#001C7F", "#12711C", "#8C0800", "#591E71", "#B8850A", "#006374"],
    colorblind=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"],
    colorblind6=["#0173B2", "#029E73", "#D55E00", "#CC78BC", "#ECE133", "#56B4E9"],
    blues=['#dbe9f6', '#bad6eb', '#89bedc', '#539ecd', '#2b7bba', '#0b559f']
)


class SeabornPaletteNames:
    deep = "deep"
    deep6 = "deep6"
    muted = "muted"
    muted6 = "muted6"
    pastel = "pastel"
    pastel6 = "pastel6"
    bright = "bright"
    bright6 = "bright6"
    dark = "dark"
    dark6 = "dark6"
    blues = "blues"

    @staticmethod
    def get_seaborn_palette_names_list():
        seaborne_palette_names = []
        for key in SEABORN_PALETTES:
            seaborne_palette_names.append(key)
        return seaborne_palette_names


class SeabornHexPalettes:
    deep = SEABORN_PALETTES["deep"]
    deep6 = SEABORN_PALETTES["deep6"]
    muted = SEABORN_PALETTES["muted"]
    muted6 = SEABORN_PALETTES["muted6"]
    pastel = SEABORN_PALETTES["pastel"]
    pastel6 = SEABORN_PALETTES["pastel6"]
    bright = SEABORN_PALETTES["bright"]
    bright6 = SEABORN_PALETTES["bright6"]
    dark = SEABORN_PALETTES["dark"]
    dark6 = SEABORN_PALETTES["dark6"]
    colorblind = SEABORN_PALETTES["colorblind"]
    colorblind6 = SEABORN_PALETTES["colorblind6"]
    blues = SEABORN_PALETTES["blues"]

    @staticmethod
    def get_palette_by_name(name):
        return SEABORN_PALETTES[name]


class AllColorPalettes:
    seaborn = SeabornHexPalettes()
    seaborn_palette_names = SeabornPaletteNames()

    @staticmethod
    def get_complete_color_palette_list():
        complete_list = ["seaborn_deep",
                         "seaborn_muted",
                         "seaborn_pastel",
                         "seaborn_bright",
                         "seaborn_dark",
                         "seaborn_colorblind"]
        return complete_list

    @staticmethod
    def get_rgb_palette_by_name(name):
        stop = 1
