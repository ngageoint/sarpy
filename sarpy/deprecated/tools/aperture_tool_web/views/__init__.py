from flask import Blueprint


mb = Blueprint(
    'aperture_tool',
    __name__,
    template_folder='../templates',
    static_folder='../static',
    static_url_path='/aperture_tool/static'
)
