# import awkward as ak
import logging

import awkward as ak
import dask_awkward as dak
import hist.dask as hda

logger = logging.getLogger(__name__)



class HistogramBuilder:
    def __init__(self, event_weights, category_axes=None, category_values=None):
        self.cat_axes = list(category_axes or [])
        self.cat_values = list(category_values or [])
        self.event_weights = event_weights

    def addCategory(self, axis, value):
        logger.debug(f"Adding axis with name {axis.name} and values {value} to histogram builder.")
        self.cat_axes.append(axis)
        self.cat_values.append(value)

    def fillHistogram(self, histogram, data, mask=None, auto_expand=True):
        logger.debug(f"Filling histogram {histogram.name} with data {data} and mask {mask}")
        weights = self.event_weights.weight()
        if not isinstance(data, list):
            data = [data]
        if not data:
            raise Exception("No data")
        h = histogram
        if weights is not None:
            if isinstance(weights, (ak.Array, dak.Array)):
                weights = (
                    weights[mask] if mask is not None else weights
                )
            else:
                weights = self.event_weights

        base_category_vals = self.cat_values
        if mask is not None:
            base_category_vals = [
                x[mask] if isinstance(x, (ak.Array, dak.Array)) else x
                for x in base_category_vals
            ]
        shaped_cat_vals = base_category_vals
        shaped_data_vals = data
        if auto_expand:
            mind, maxd = data[0].layout.minmax_depth
            if maxd > 1:
                ol = ak.ones_like(data[0])
                if weights is not None:
                    weights = ak.flatten(ol * weights)
                shaped_cat_vals = [
                    ak.flatten(ol * x) if isinstance(x, (ak.Array, dak.Array)) else x
                    for x in base_category_vals
                ]
                shaped_data_vals = [
                    ak.flatten(x) if isinstance(x, (ak.Array, dak.Array)) else x
                    for x in shaped_data_vals
                ]
        logger.debug(f"Shaped category values are\n {shaped_cat_vals}")
        logger.debug(f"Shaped data values are\n {shaped_data_vals}")
        d = shaped_cat_vals + shaped_data_vals
        ret = h.fill(*d, weight=weights)
        return ret

    def createHistogram(
        self,
        axis,
        name=None,
        description=None,
    ):
        logger.debug(f"Creating histogram with axes {axis} and name {name}")

        if isinstance(axis, list):
            all_axes = self.cat_axes + list(axis)
        else:
            all_axes = self.cat_axes + [axis]

        h = hda.Hist(*all_axes, storage="weight", name=name)
        setattr(h, "description", description)
        setattr(h, "name", name)
        return h

    def __call__(self, *args, **kwargs):
        return self.makeHistogram(*args, **kwargs)
