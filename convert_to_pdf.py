import comtypes.client
import os
import glob
import time

folder = r"C:\Users\Immanuelle\Downloads\notebookLM"
pptm_files = glob.glob(os.path.join(folder, "*.pptm"))

print(f"Found {len(pptm_files)} .pptm files to convert")

powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
powerpoint.Visible = 1

for pptm_path in pptm_files:
    pdf_path = pptm_path.replace(".pptm", ".pdf")
    basename = os.path.basename(pptm_path)
    try:
        presentation = powerpoint.Presentations.Open(pptm_path, WithWindow=False)
        # 32 = ppSaveAsPDF
        presentation.SaveAs(pdf_path, 32)
        presentation.Close()
        print(f"OK: {basename}")
    except Exception as e:
        print(f"FAIL: {basename} - {e}")
    time.sleep(0.5)

powerpoint.Quit()
print("\nDone!")
