from flask import Flask, request
from markupsafe import escape
from transformers import AutoTokenizer, PhiForCausalLM

app = Flask(__name__,)

tokenizer = AutoTokenizer.from_pretrained("phi-2", torch_dtype="auto", trust_remote_code=True, torch_device='cpu')
model = PhiForCausalLM.from_pretrained("phi-2",trust_remote_code=True, local_files_only=True)

@app.route("/")
def hello_world():
    return "<p>server up!</p>"

@app.route("/<name>")
def name(name):
    return f"<p>{escape(name)}!</p>"

#?key=value
@app.route("/log")
def login():
    #text = "Hello, how are you?"
    text = request.args.get('key','')
    prompt = "Instruct:  You are a helpful assistant. I have a question for you that i want you to give a well thought out answer to. My question is, "+text+"\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt",return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=300)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(generated_text)
    #user = request.args.get('key','')
    return f"<p>{escape(generated_text)}</p>"

'''
@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'],
                       request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password'
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login.html', error=error)
'''