"""Non-promoted V17.7 four-material coverage and semantic mip reconstruction.

V5 material *identity* remains a discrete native-resolution label. Smooth
fractions at enhanced resolution represent pixel coverage of up to four
labels, not an interpolated scalar selector. Materials not using the V5
family keep their original shader paths. All images are BGRA in OpenCV.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

CENTRES = np.array([0.0, 1/3, 2/3, 1.0],dtype=np.float32)
SCHEMA = "NSAMDR_V177_SUBPIXEL_MATERIAL_COVERAGE_V1"


def _quantize_selector(selector: np.ndarray) -> np.ndarray:
    if selector.ndim != 2:
        raise ValueError("selector must be a native-resolution H×W plane")
    value=selector.astype(np.float32)
    if np.issubdtype(selector.dtype,np.integer):
        value=value/255.
    if not np.isfinite(value).all() or value.min()<0 or value.max()>1:
        raise ValueError("selector contains values outside [0,1]")
    return np.argmin(np.abs(value[...,None]-CENTRES),axis=2).astype(np.uint8)


def _normalized(weights: np.ndarray) -> np.ndarray:
    weights=np.maximum(weights,0).astype(np.float32)
    total=np.sum(weights,axis=2,keepdims=True)
    # The label set is exhaustive so total must never reach zero.
    if (total<=0).any():
        raise RuntimeError("material coverage contains an unassigned pixel")
    return weights/total


def rgba_coverage(selector: np.ndarray, *, scale: int=4, smoothing: float=.70) -> np.ndarray:
    """Return H×W×4 uint8 *RGBA*, with finite normalized class coverage.

    Signed distances are measured in native pixels. A low-radius Gaussian on
    the signed field regularises one-pixel raster stair-steps, while zero
    weights for absent classes and a conservative width protect identity.
    This is a geometrical estimate, not an inference of authored 3D edges.
    """
    if scale!=4 or not (0<=smoothing<=1.25):
        raise ValueError("native V17.7 coverage requires 4x and bounded smoothing")
    classes=_quantize_selector(selector)
    h,w=classes.shape
    if min(h,w)<2:
        raise ValueError("selector source must be at least 2×2")
    size=(w*scale,h*scale)
    present=np.unique(classes).tolist()
    fields=[]
    for category in range(4):
        if category not in present:
            fields.append(np.zeros((size[1],size[0]),dtype=np.float32))
            continue
        if len(present)==1:
            fields.append(np.ones((size[1],size[0]),dtype=np.float32))
            continue
        mask=np.uint8(classes==category)
        inside=cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        outside=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        # Signed distances corrected to edge placement between pixel centres.
        distance=np.where(mask!=0,inside-.5,-outside+.5)
        if smoothing>0:
            distance=cv2.GaussianBlur(
                distance,(0,0),sigmaX=smoothing,sigmaY=smoothing,
                borderType=cv2.BORDER_REFLECT_101,
            )
        subpixel=cv2.resize(distance,size,interpolation=cv2.INTER_CUBIC)
        fields.append(np.clip(.5+subpixel/.8,0,1).astype(np.float32))
    coverage=_normalized(np.stack(fields,axis=2))
    result=np.rint(coverage*255).astype(np.uint8)
    # Keep the class sum equal to 255 after quantisation, without modifying
    # absent slots or changing the dominant identity in uniform interiors.
    error=255-np.sum(result.astype(np.int16),axis=2)
    largest=np.argmax(coverage,axis=2)
    ys,xs=np.indices(classes.shape)*0 if False else np.indices((h*scale,w*scale))
    adjusted=result[ys,xs,largest].astype(np.int16)+error
    result[ys,xs,largest]=np.clip(adjusted,0,255).astype(np.uint8)
    if not np.all(np.sum(result.astype(np.int16),axis=2)==255):
        raise RuntimeError("subpixel coverage quantization failed")
    return result


def weights_to_bgra(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim!=3 or rgba.shape[2]!=4 or rgba.dtype!=np.uint8:
        raise ValueError("coverage texture must be RGBA8")
    return rgba[:,:,[2,1,0,3]].copy()


def write_coverage(selector: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    bgra=weights_to_bgra(rgba_coverage(selector))
    if not cv2.imwrite(str(destination),bgra,[cv2.IMWRITE_PNG_COMPRESSION,3]):
        raise RuntimeError(f"failed to save coverage: {destination}")


def semantic_half(image: np.ndarray, *, mode: str,
                  selector_channel: int | None=None,
                  normal_channels: tuple[int,int] | None=None) -> np.ndarray:
    """Downsample BGRA8 with semantic treatment of signed normals and IDs."""
    if image.ndim!=3 or image.shape[2]!=4 or image.dtype!=np.uint8:
        raise ValueError("semantic mips require BGRA8")
    h,w=image.shape[:2]
    if h==1 and w==1:
        return image.copy()
    size=(max(1,w//2),max(1,h//2))
    if mode=="coverage":
        rgba=image[:,:,[2,1,0,3]].astype(np.float32)/255.
        downs=cv2.resize(rgba,size,interpolation=cv2.INTER_AREA)
        downs=_normalized(downs)
        result=np.rint(downs*255).astype(np.uint8)
        error=255-result.astype(np.int16).sum(2)
        inds=np.argmax(downs,axis=2)
        yy,xx=np.indices(inds.shape)
        result[yy,xx,inds]=np.clip(result[yy,xx,inds].astype(np.int16)+error,0,255)
        return weights_to_bgra(result)
    if mode=="categorical":
        result=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
        if selector_channel is not None:
            result[:,:,selector_channel]=cv2.resize(
                image[:,:,selector_channel],size,interpolation=cv2.INTER_NEAREST,
            )
        return result
    if mode=="normal":
        if normal_channels is None:
            raise ValueError("normal mip requires signed XY channel indices")
        c0,c1=normal_channels
        if c0==c1 or min(c0,c1)<0 or max(c0,c1)>3:
            raise ValueError("invalid normal map channels")
        # Preserve other packed channels, but reconstruct signed XY jointly.
        result=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
        x=result[:,:,c0].astype(np.float32)/127.5-1
        y=result[:,:,c1].astype(np.float32)/127.5-1
        length=np.maximum(1,np.hypot(x,y)/.999)
        result[:,:,c0]=np.rint(np.clip((x/length+1)*127.5,0,255)).astype(np.uint8)
        result[:,:,c1]=np.rint(np.clip((y/length+1)*127.5,0,255)).astype(np.uint8)
        return result
    if mode=="albedo":
        # Decode and encode sRGB: data averaging in gamma-encoded space
        # artificially darkens edge antialiasing in the DX11 sRGB loader.
        bgr=image[:,:,:3].astype(np.float32)/255
        linear=np.where(bgr<=.04045,bgr/12.92,((bgr+.055)/1.055)**2.4)
        down=cv2.resize(linear,size,interpolation=cv2.INTER_AREA)
        enc=np.where(down<=.0031308,12.92*down,1.055*np.power(np.maximum(down,0),1/2.4)-.055)
        alpha=cv2.resize(image[:,:,3],size,interpolation=cv2.INTER_AREA)
        return np.dstack((
            np.rint(np.clip(enc*255,0,255)).astype(np.uint8),alpha
        ))
    if mode=="continuous":
        return cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    raise ValueError(f"unsupported semantic mip type: {mode}")


def write_semantic_mips(path: Path, *, mode: str,
                        selector_channel: int | None=None,
                        normal_channels: tuple[int,int] | None=None) -> int:
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=3 or image.shape[2]!=4:
        raise RuntimeError(f"cannot read BGRA semantic source: {path}")
    levels=0
    current=image
    while current.shape[0]>1 or current.shape[1]>1:
        current=semantic_half(current,mode=mode,selector_channel=selector_channel,
                              normal_channels=normal_channels)
        levels+=1
        dest=path.with_name(f"{path.stem}.mip{levels}{path.suffix}")
        if not cv2.imwrite(str(dest),current,[cv2.IMWRITE_PNG_COMPRESSION,3]):
            raise RuntimeError(f"cannot save semantic mip: {dest}")
    return levels
