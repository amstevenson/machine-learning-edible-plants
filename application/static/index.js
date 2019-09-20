ENDPOINT = "http://localhost:5000";

submit = function() {
  text = document.getElementById('text_input').value;
  console.log('testing')
  var xhttp = new XMLHttpRequest();
  xhttp.addEventListener("readystatechange", function () {
    if (this.readyState === 4) {
      document.getElementById("result").innerHTML = "RESULT: "+this.responseText;
    }
  });
  xhttp.open("GET", ENDPOINT + "/?input=" + text, true);
  xhttp.send();
};

imgSize = function() {
  text = document.getElementById('url_input').value;
  console.log('testing')
  var xhttp = new XMLHttpRequest();
  xhttp.addEventListener("readystatechange", function () {
    if (this.readyState === 4) {
      alert("Backend says the image has - " + this.responseText);
    }
  });
  xhttp.open("GET", ENDPOINT + "/imgSize/?input=" + text, true);
  xhttp.send();
};

healthCheck = function() {
  text = document.getElementById('text_input').value;
  console.log('testing')
  var xhttp = new XMLHttpRequest();
  xhttp.addEventListener("readystatechange", function () {
    if (this.readyState === 4) {
      alert(this.responseText);
    }
  });
  xhttp.open("GET", ENDPOINT + "/healthcheck", true);
  xhttp.send();
};

imgUpload = function() {
  text = document.getElementById('text_input').value;
  console.log('testing')
  var xhttp = new XMLHttpRequest();
  xhttp.addEventListener("readystatechange", function () {
    if (this.readyState === 4) {
      if (this.responseText == "ok") {
        alert("Image successfully uploaded!");
      } else {
        alert("Something went wrong!");
        console.log(this.responseText)
      }
    }
  });
  var fileInput = document.getElementById('img_upload');
  var file = fileInput.files[0];
  var formData = new FormData();
  formData.append('file', file);
  xhttp.open("POST", ENDPOINT + "/imgUpload", true);
  xhttp.send(formData);
};
