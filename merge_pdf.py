from PyPDF2 import PdfMerger

pdfs = ['1.pdf', '2.pdf', '3.pdf', '4.pdf']

merger = PdfMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write("result.pdf")
merger.close()

