import cv2
import numpy as np
import pywt

EMBEDDING_STRENGTH = 0.05

def apply_fwt(img):
    coeffs = pywt.dwt2(img, 'haar')
    return coeffs

def inverse_fwt(coeffs):
    return pywt.idwt2(coeffs, 'haar')

def embed_watermark(cover_img_path, watermark_img_path, output_path):
    cover_img = cv2.imread(cover_img_path)
    watermark_img = cv2.imread(watermark_img_path, cv2.IMREAD_GRAYSCALE)

    cover_img = cv2.resize(cover_img, (256, 256))
    watermark_img = cv2.resize(watermark_img, (256, 256))

    cover_ycrcb = cv2.cvtColor(cover_img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(cover_ycrcb)

    LL, (LH, HL, HH) = apply_fwt(Y)
    _, (WM_LH, WM_HL, WM_HH) = apply_fwt(watermark_img)

    LH += EMBEDDING_STRENGTH * WM_LH
    HL += EMBEDDING_STRENGTH * WM_HL
    HH += EMBEDDING_STRENGTH * WM_HH

    Y_embedded = inverse_fwt((LL, (LH, HL, HH)))
    Y_embedded = np.uint8(Y_embedded)

    embedded_img = cv2.merge([Y_embedded, Cr, Cb])
    embedded_img = cv2.cvtColor(embedded_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, embedded_img)

def extract_watermark(embedded_img_path):
    embedded_img = cv2.imread(embedded_img_path)
    embedded_ycrcb = cv2.cvtColor(embedded_img, cv2.COLOR_BGR2YCrCb)
    Y_embedded, _, _ = cv2.split(embedded_ycrcb)

    LL, (LH, HL, HH) = apply_fwt(Y_embedded)

    extracted_LH = LH / EMBEDDING_STRENGTH
    extracted_HL = HL / EMBEDDING_STRENGTH
    extracted_HH = HH / EMBEDDING_STRENGTH

    extracted_watermark = inverse_fwt((np.zeros_like(LL), (extracted_LH, extracted_HL, extracted_HH)))
    extracted_watermark = cv2.normalize(extracted_watermark, None, 0, 255, cv2.NORM_MINMAX)
    extracted_watermark = np.uint8(extracted_watermark)

    _, extracted_bw = cv2.threshold(extracted_watermark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return extracted_bw
