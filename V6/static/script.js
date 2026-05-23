async function sendMessage(){

    let input = document.getElementById("message");
    let message = input.value;

    if(!message) return;

    let chatBox = document.getElementById("chat-box");

    chatBox.innerHTML += `
        <div class="user">
            <b>You:</b> ${message}
        </div>
    `;

    input.value = "";

    let response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type":"application/json"
        },
        body: JSON.stringify({
            message: message
        })
    });

    let data = await response.json();

    chatBox.innerHTML += `
        <div class="bot">
            <b>Bot:</b> ${data.reply}
        </div>
    `;

    chatBox.scrollTop = chatBox.scrollHeight;
}