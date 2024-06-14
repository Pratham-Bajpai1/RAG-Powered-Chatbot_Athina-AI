import PyPDF2

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    pdf_file = open(file_path, "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
    pdf_file.close()
    return pdf_text

# Extract text from the provided PDF
pdf_file_path = "policy-booklet-0923.pdf"
pdf_text = extract_text_from_pdf(pdf_file_path)

with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(pdf_text)
print("Text extraction completed.")

# Displaying a portion of the extracted text
print(pdf_text[:2000])  # Displaying the first 2000 characters for brevity