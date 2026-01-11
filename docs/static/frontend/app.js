const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const output = document.getElementById("chat-output");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const userMessage = input.value;
    output.innerHTML += `<p><b>You:</b> ${userMessage}</p>`;

    try {
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: userMessage })
        });
        const data = await response.json();
        output.innerHTML += `<p><b>Bot:</b> ${data.answer}</p>`;
    } catch (err) {
        output.innerHTML += `<p style="color:red;">Error: ${err}</p>`;
    }

    input.value = "";
});
