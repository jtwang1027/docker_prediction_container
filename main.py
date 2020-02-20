from flask import Flask, request, render_template, request, redirect 
import pandas as pd
app = Flask(__name__)


@app.route('/upload-image', methods=['GET', 'POST']) #get/post generates a request object
def upload_image():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            print(image)
            
            return redirect(request.url)

    return render_template('upload_image.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
