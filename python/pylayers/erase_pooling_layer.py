import _init_paths
import caffe
import numpy as np
import yaml
import six


class ErasePoolingLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # Check input pair
        if len(bottom) != 3:
            raise Exception("Need three input: FeatMap, HeatMap, cls.")
        # Load layer parameter
        layer_params = yaml.load(self.param_str)
        self._erase_ratio = layer_params.get('erase_ratio', 0)
        self._erase_thresh = layer_params.get('erase_thresh', -1)

    def reshape(self, bottom, top):
        # Check input shape match
        if bottom[0].num != bottom[1].num or bottom[1].num != bottom[2].num:
            raise Exception("Nums of FeatMap, HeatMap and cls not match.")
        if bottom[0].data.shape[2:] != bottom[1].data.shape[2:]:
            raise Exception("Spatial shape of FeatMap and HeatMap not match.")
        if len(bottom[2].data.shape) != 2:
            raise Exception("Shape of cls should be (N,C).")
        # Shape of output is the same as FeatMap
        top[0].reshape(*bottom[0].shape)
        # Reshape heatmap state
        self._erase_maps = []

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data[...]
        clsIdx = np.argmax(bottom[2].data, axis=1)
        for idx,cls in enumerate(clsIdx):
            prob_map = bottom[1].data[idx,cls]
            heat_array = prob_map.flatten()
            if self._erase_thresh <= 0:
                thresh_ind = int(self._erase_ratio * heat_array.size)
                self._erase_thresh = np.argsort(heat_array)[::-1][thresh_ind]
            erase_map = prob_map > self._erase_thresh
            top[0].data[idx,:, erase_map] = 0
            self._erase_maps.append(erase_map)

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff[...]
        for idx in six.moves.range(bottom[0].num):
            erase_map = self._erase_maps[idx]
            bottom[0].diff[idx,:, erase_map] = 0
