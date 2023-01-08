# patch-free
input_img_size=(64,96,96)
crop_size=input_img_size

# patch-based (64,96,96). The model input is LABEL_SR, so it's actually doing (64,96,96) patch on (128,192,192) image
# input_img_size=(64,96,96)
# crop_size=(32,48,48) # crop image will not be used in this experiment, instead the (64,96,96) label_sr gt will be used.

# # patch-based (64,64,64)
# input_img_size=(64,96,96)
# crop_size=(64,64,64)