from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    packet_rate = float(request.form['packet_rate'])
    failed_logins = float(request.form['failed_logins'])
    traffic_load = float(request.form['traffic_load'])

    score = packet_rate + failed_logins + traffic_load

    if score > 80:
        result = "Intrusion Detected"
    else:
        result = "Normal Traffic"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
