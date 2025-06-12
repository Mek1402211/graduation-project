import pytesseract
from googletrans import Translator
import cv2

async def ocr_and_translate(image_path):
    # Preprocess image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # OCR
    text = pytesseract.image_to_string(thresh, lang='eng+spa+fra+ar')
    
    # Translate to Arabic
    translator = Translator()
    translat=await translator.translate(text, dest="ar")#translate the text to the target language using googletrans
    return translat.text


