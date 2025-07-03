from flask import Flask
from apis.OCR_CNPJ_PDF import ocr_cnpj_api_bp

app = Flask(__name__)
app.register_blueprint(ocr_cnpj_api_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
