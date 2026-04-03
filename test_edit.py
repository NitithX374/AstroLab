import sys, io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, '.')
from astrolab.cli.parser import AstroLabCLI

cli = AstroLabCLI()

print("=" * 55)
print("  edit body — end-to-end test")
print("=" * 55)

print("\n1. Create 'tau' without radius:")
cli.onecmd("create body tau mass=1.989e30 pos=(0,0,0) type=star")

print("\n2. Show before edit:")
cli.onecmd("show body tau")

print("\n3. Fix missing radius:")
cli.onecmd("edit body tau radius=695700000")
cli.onecmd("show body tau")

print("\n4. Edit multiple fields at once:")
cli.onecmd("edit body tau type=star color=orange mass=2.1e30")
cli.onecmd("show body tau")

print("\n5. Bad body name:")
cli.onecmd("edit body mars radius=3389500")

print("\n6. No fields supplied — should warn:")
cli.onecmd("edit body tau")

print("\n7. Unknown field — should warn:")
cli.onecmd("edit body tau brightness=9000")

print("\n" + "=" * 55)
print("  All edit tests done.")
print("=" * 55)
