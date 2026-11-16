"""Extract the bundled paper data into a new directory."""
from pathlib import Path
import sys,tarfile,tempfile
archive,destination=map(Path,sys.argv[1:])
if destination.exists():raise FileExistsError(destination)
with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
 with tarfile.open(archive) as data:
  members=data.getmembers()
  for member in members:
   p=Path(member.name)
   if p.is_absolute() or '..' in p.parts or p.parts[0]!='bipga' or not (member.isfile() or member.isdir()):raise ValueError('Unsafe archive member')
  data.extractall(temporary,members=members)
 (Path(temporary)/'bipga').rename(destination)
