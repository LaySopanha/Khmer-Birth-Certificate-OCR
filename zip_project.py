import shutil

# Archive the entire folder
shutil.make_archive("/teamspace/studios/this_studio/output_export", 'zip', "/teamspace/studios/this_studio")
shutil.move("/teamspace/studios/this_studio/output_export.zip", "/app/outputs/output_export.zip")
