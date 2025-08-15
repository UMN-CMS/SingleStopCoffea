import mplhep

from .common import PlotConfiguration


def addCMSBits(
    ax,
    sectors,
    extra_text=None,
    text_color=None,
    plot_configuration=None,
):
    if plot_configuration is None:
        plot_configuration = PlotConfiguration()
    info_text = plot_configuration.lumi_text
    if info_text is None:
        lumis = set(str(x.dataset.lumi) for x in sectors)
        energies = set(str(x.dataset.era.energy) for x in sectors)
        era = set(str(x.dataset.era.name) for x in sectors)
        era_text = f"{'/'.join(era)}"
        if "dijet" in era_text:
            era_text = '2018'
        lumi_text = (
            plot_configuration.lumi_text
            or f"{'/'.join(lumis)} fb$^{{-1}}$ ({'/'.join(energies)} TeV)"
        )
        info_text = era_text + ", " + lumi_text
    mplhep.cms.lumitext(text=info_text, ax=ax)

    text = plot_configuration.cms_text
    if extra_text is not None:
        text += "\n" + extra_text
    a, b, c = mplhep.cms.text(text=text, ax=ax, loc=plot_configuration.cms_text_pos)
    if text_color is not None:
        a.set(color=plot_configuration.cms_text_color)
        b.set(color=plot_configuration.cms_text_color)
        c.set(color=plot_configuration.cms_text_color)


def labelAxis(ax, which, axes, label=None, label_complete=None):
    mapping = dict(x=0, y=1, z=2)
    idx = mapping[which]

    if idx != len(axes):
        this_unit = getattr(axes[idx], "unit", None)
        if not label:
            label = axes[idx].label
            if this_unit:
                label += f" [{this_unit}]"
        getattr(ax, f"set_{which}label")(label.replace("textrm", "text"))
    else:
        label = label or "Events"
        units = [getattr(x, "unit", None) for x in axes]
        units = [x for x in units if x]
        unit_format = "*".join(units)
        if unit_format:
            label += f" / {unit_format}"
        getattr(ax, f"set_{which}label")(label.replace("textrm", "text"))
