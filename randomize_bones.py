import bpy
import json

measurements_file = r"C:\Users\PC\Desktop\sdsds\g67Backend-main1\normalized_measurements.txt"
try:
    with open(measurements_file, 'r') as file:
        measurements = json.load(file)
        print("Loaded normalized measurements:", measurements)
except Exception as e:
    print(f"Failed to load measurements from file: {e}")
    measurements = {}

armature = bpy.data.objects.get('Adjustable Mannequin')
# Apply measurements to bones
for bone_name, value in measurements.items():
    bone = bpy.data.objects['Adjustable Mannequin'].pose.bones.get(bone_name)
    if bone:
        bone.location.x = value  # Assuming x-axis adjustment
        print(f"Set '{bone.name}' location.x to {value:.3f}")
    else:
        print(f"Bone '{bone_name}' not found.")

# Adjust the head style
bone = armature.pose.bones.get("head style")
if bone:
    bone.location.x = 0.57
    print("Set 'head style' location.x to 0.57")
else:
    print("Bone 'head style' not found.")

# Adjust the root bone location
root_bone = armature.pose.bones.get('root')
if root_bone:
    root_bone.location.y += -3.5  # Move the bone along the Y-axis
    root_bone.location.x += -0.8  # Move the bone along the X-axis
    print(f"Adjusted root bone to X: {root_bone.location.x:.3f}, Y: {root_bone.location.y:.3f}")
else:
    print("Root bone not found.")

# Update the scene
bpy.context.view_layer.update()

# Camera setup
camera = bpy.data.objects['Camera']
model = bpy.data.objects['Adjustable Mannequin']

# Move the camera further back to zoom out
direction = model.location - camera.location  # Direction from camera to model
distance = 13.0  # Adjust this value to zoom out (higher = more zoomed out)
camera.location = model.location + direction.normalized() * -distance

# Adjust camera orientation to point at the model
camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# Set camera as the active camera
bpy.context.scene.camera = camera

# Adjust render resolution
bpy.context.scene.render.resolution_x = \
    0
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.render.resolution_percentage = 100

# Render and save the image
image_output_path = "C:/Users/PC/Downloads/Adjusted_Mannequin_YAdjusted.png"
bpy.context.scene.render.filepath = image_output_path
bpy.ops.render.render(write_still=True)
print(f"Snapshot saved to {image_output_path}")

obj_output_path = r"C:\Users\PC\Desktop\sdsds\g67Backend-main1\Adjusted_Mannequin.obj"

# Ensure the mannequin, shirt, and related objects are selected for export
bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
model.select_set(True)  # Select the mannequin model
armature.select_set(True)  # Select the armature

# Find and select the shirt object
shirt = bpy.data.objects.get('Shirt OBJ')  # Replace 'Shirt' with the actual name of your shirt object
if shirt:
    shirt.select_set(True)
    shirt.location.x += -0.8
    shirt.location.y += -3.5
    print("Shirt object found and selected for export.")
else:
    print("Shirt object not found. Please check the object name.")

# Set the selected objects as the active context
bpy.context.view_layer.objects.active = model

# Use the correct export operator and apply necessary transformations
bpy.ops.wm.obj_export(
    filepath=obj_output_path
)

print(f"Model exported as OBJ to {obj_output_path}")
