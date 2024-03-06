# DUA in Semantic Segmentation Networks
Implementation of DUA: Dynamic Unsupervised Adaption for Semantic Segmentation Networks. DUA is a method for unsupervised domain adaption. DUA was originally developed by Mirza et al. [Official Repository](https://github.com/jmiemirza/DUA). 
This project implements DUA in Semantic Segmentation Networks. Models were initially trained on Cityscapes and are adapted via DUA on the ACDC dataset. Modelarchitecture is based on code by [Gongfan Fang](https://github.com/VainF/DeepLabV3Plus-Pytorch/tree/master).

# Examples

## 1. Cityscapes Predictions (base model)
72% (mIoU) on CS dataset
<div>
<img src="samples/test_image_CS.png"   width="30%">
<img src="samples/test_target_CS.png"    width="30%">
<img src="samples/test_pred_CS.png"  width="30%">
</div>

## 2. ACDC Rain
<div>
<img src="samples/test_image_rain.png"   width="30%">
<img src="samples/test_target_rain_7.png"    width="30%">
</div>
(before and after adaptation)
<div>
 <img src="samples/test_pred_rain_base.png" width="30%"/>
<img src="samples/test_pred_rain_adapted.png" width="30%" />
</div>

## 3. ACDC Snow
<div>
<img src="samples/test_image_snow_1.png"   width="30%">
<img src="samples/test_target_snow_1.png"    width="30%">
</div>
(before and after adaptation)
<div>
 <img src="samples/test_pred_snow_base_1.png" width="30%"/>
<img src="samples/test_pred_snow_adapted_1.png" width="30%" />
</div>

