from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    stats = {'total_vectors': 0}
    return render_template('index.html', stats=stats)

if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📖 Open: http://localhost:5000")
    app.run(debug=True, port=5000)
