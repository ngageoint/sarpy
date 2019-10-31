from flask import Blueprint


mb = Blueprint(
    'taser',
    __name__,
    template_folder='../templates',
    static_folder='../static',
    # static_url_path='../static'
)
