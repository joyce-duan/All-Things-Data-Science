'''
source: http://stackoverflow.com/questions/5725278/python-help-using-pdfminer-as-a-library
'''

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import os
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = file(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    str = retstr.getvalue()
    retstr.close()
    return str

if __name__ == '__main__':
    #fname = '/Users/joyceduan/data/slack/06-04.pdf'
    mydirectory = '/Users/joyceduan/data/slack/'
    #mydirectory = '/Users/joyceduan/documents/git/All-Things-Data-Science/code/data/'
    fnames = os.listdir(mydirectory)
    for fname in fnames:
        if fname[-4:] == '.pdf':
            fname_full = mydirectory + fname
            print fname_full
            txt = convert_pdf_to_txt(fname_full)
            out_fname = fname.replace('.pdf','.txt')
            with open(out_fname, 'w') as out_fh:
                out_fh.write(txt+"\n")
