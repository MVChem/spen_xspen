"""Render the actual final PPT via LibreOffice and Poppler."""
from pathlib import Path
import subprocess,tempfile
here=Path(__file__).resolve().parent
out=here.parents[1]/'ppt'
with tempfile.TemporaryDirectory(prefix='fig1_v2_') as tmp:
    temp=Path(tmp)
    subprocess.run(['libreoffice',f'-env:UserInstallation={(temp/"profile").as_uri()}','--headless','--convert-to','pdf','--outdir',tmp,str(out/'fig1_v2.pptx')],check=True)
    subprocess.run(['pdftoppm','-scale-to','1440','-png','-singlefile',str(temp/'fig1_v2.pdf'),str(out/'fig1_v2')],check=True)
