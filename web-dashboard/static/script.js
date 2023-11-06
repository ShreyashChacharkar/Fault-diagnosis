const body = document.querySelector("body"),
      modeToggle = body.querySelector(".mode-toggle");
      sidebar = body.querySelector("nav");
      sidebarToggle = body.querySelector(".sidebar-toggle");

let getMode = localStorage.getItem("mode");
if(getMode && getMode ==="dark"){
    body.classList.toggle("dark");
}

let getStatus = localStorage.getItem("status");
if(getStatus && getStatus ==="close"){
    sidebar.classList.toggle("close");
}

modeToggle.addEventListener("click", () =>{
    body.classList.toggle("dark");
    if(body.classList.contains("dark")){
        localStorage.setItem("mode", "dark");
    }else{
        localStorage.setItem("mode", "light");
    }
});

sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("close");
    if(sidebar.classList.contains("close")){
        localStorage.setItem("status", "close");
    }else{
        localStorage.setItem("status", "open");
    }
})


$(document).ready(function() {
    $('a[href^="#"]').on('click', function(event) {
        event.preventDefault();
        let target = $(this).attr('href');
        $('html, body').animate({
            scrollTop: $(target).offset().top
        }, 800);
    });
});


function predictFunction(button) {
    var row = button.closest('tr');
    var cells = row.querySelectorAll('td');
    var rowData = [];
    for (var i = 0; i < cells.length; i++) {
        rowData.push(cells[i].textContent.trim());
    }

    // Set the input_data field value
    document.querySelector('input[name="input_data"]').value = JSON.stringify(rowData);

    // Send an AJAX request to the server
    $.ajax({
        type: 'POST',
        url: '/submit',
        data: $('form').serialize(),
        success: function(response) {
            // Handle the response from the server (e.g., update the table with prediction results)
            $('.result').html("Prediction Result: " + response);
        },
        error: function(error) {
            // Handle errors, if any
            $('.result').html("Error: " + error);
        }
    });
}
// Include jQuery library for AJAX

const radioButtons = document.querySelectorAll('input[type="radio"]');
        
radioButtons.forEach(radioButton => {
    radioButton.addEventListener('change', function() {
        const selectedOption = this.value;
        fetch('/update_option', {
            method: 'POST',
            body: JSON.stringify({ selectedOption }),
            headers: {
                'Content-Type': 'application/json'
            }
        });
    });
})




