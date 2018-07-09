import numpy as np
import theano

def shape_data(original_data):
    m,r,c = original_data.shape
    data = original_data.reshape((m,r*c))
    shared_data = theano.shared(np.asarray(data,dtype=theano.config.floatX),borrow=True)
    return shared_data

def scale_to_unit_interval(ndar, eps=1e-8):
    '''Scales all values in the ndarray ndar to be between 0 and 1'''
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X,img_shape,tile_shape,tile_spacing=(0, 0),scale_rows_to_unit_interval=True,output_pixel_vals=True):
    assert len(img_shape) == 2;assert len(tile_shape) == 2;assert len(tile_spacing) == 2
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),dtype=X.dtype)

        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,dtype=dt) + channel_defaults[i]
            else:
                out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        H, W = img_shape
        Hs, Ws = tile_spacing
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        this_img = scale_to_unit_interval(this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[tile_row * (H + Hs): tile_row * (H + Hs) + H,tile_col * (W + Ws): tile_col * (W + Ws) + W] = this_img * c
        return out_array