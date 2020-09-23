import fitz
import sys
from PIL import Image

doc = fitz.open('TensorFlow_Machine_Learning_Cookbook.pdf')

for pg in range(doc.pageCount):
    page = doc[pg]
    zoom = int(1000)
    rotate = int(0)
    trans = fitz.Matrix(zoom / 100.0, zoom / 100.0).preRotate(rotate)

    # create raster image of page (non-transparent)
    pm = page.getPixmap(matrix=trans, alpha=False)

    # write a PNG image of the page
    # pm.writePNG('%s.png' % pg)
    # pm.writeImage("%s.gif" % pg)
    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
    img.save("%s.jpg" % pg, "JPEG")
