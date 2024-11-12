import cv2

def downscale(image, factor=2, interpolation=cv2.INTER_AREA):
    # Inter area interpolatio is the standard for downscaling
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=interpolation)
