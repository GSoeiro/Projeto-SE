import http.client
import urllib.parse
import base64

PUSHSAFER_KEY = "JNhGE1z7zxT1dHpHJIIR"

def send_pushsafer_notification(message="Queda Detetada!",
                                title="Alerta de Queda",
                                image_path=None):

    conn = http.client.HTTPSConnection("pushsafer.com")

    image_b64 = ""

    if image_path is not None:
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                print("Base64 length:", len(image_b64))
        except Exception as e:
            print("Erro a carregar screenshot:", e)

    if image_b64 != "":
        image_b64 = "data:image/jpeg;base64," + image_b64

    payload = {
        "k": PUSHSAFER_KEY,
        "m": message,
        "t": title,
        "i": "5",
        "s": "25",
        "v": "3",
        "p": image_b64
    }

    post_data = urllib.parse.urlencode(payload)
    headers = {"Content-type": "application/x-www-form-urlencoded"}

    conn.request("POST", "/api", post_data, headers)
    response = conn.getresponse()

    print("PushSafer response:", response.status, response.reason)
    print(response.read().decode())
