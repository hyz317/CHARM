import bpy
import sys

def install_addon(addon_path):
    bpy.ops.preferences.addon_install(filepath=addon_path)
    bpy.ops.preferences.addon_enable(module=addon_path.split('/')[-1].replace('.py', '').replace('.zip', ''))
    bpy.ops.wm.save_userpref()
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: blender --background --python install_addon.py -- <path_to_addon>")
        sys.exit(1)

    addon_path = sys.argv[-1]
    install_addon(addon_path)