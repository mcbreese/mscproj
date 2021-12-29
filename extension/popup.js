$(document).ready(function() {
    // When the plugin is opened/ready then immediately run post-python
    console.log("popup.js loaded");
    // Initialise the url variable 
    var url;
    chrome.tabs.query({
        active: true,
        currentWindow: true
    }, function(tab) {
        // Tab is an array, position 0 is the current tab
        url = tab[0].url;
        console.log("The URL is \n" + url);
        // Measuring length of time
        var startTime = performance.now()
        postPython(url);
        var endTime = performance.now()
        console.log(`Call to doSomething took ${endTime - startTime} milliseconds`)
    });


});

// Put the popup.html into loading mode until the result is returned
function loading() {
    $("#main").removeClass("hidden");
    $("#loading").addClass("hidden");
};

// Add the classification from the model to the popup.html
function formatClassification(str) {
    $("#class").text(str);
    if (str == "REAL") {
        $(".sect1").addClass("green");
        $("#warn").addClass("hidden");
    } else {
        $(".sect1").addClass("red");
    }
};

// Add the main topics from the model to the popup.html
function formatConfidence(str) {
    // Math.round takes it to nearest int to multiply by 100 to get %
    str = Math.round(str * 100) / 100
    str *= 100
    $("#conf").text(str + "%")
};

// Add the main topics from the model to the popup.html
function formatTopics(arr) {
    topicOut = []
    for (x = 0; x < arr[0].length; x++) {
        topicOut.push(" " + arr[0][x][0])
    }
    $("#topics").text(topicOut);
};

function postPython(url) {
    console.log("Model is now assessing page for misinformation");
    //url= window.location.href;
    // Regex if statement to check for google
    regex = /www.google./;
    result = url.match(regex)
    if (result) {
        console.log("This is a google search - not scraping");
        // If it fails then print error message
        str = "<h1>Please use a website, the application does not work on Google search!</p> "
        document.getElementById("loading").innerHTML = str;
    } else {
        try {
            // Try the initial block of code
            var loadURL = $.ajax({
                //url: "http://localhost:8000/cgi-bin/main.py",
                url: "http://20.117.142.62:8000/cgi-bin/main.py",
                type: "POST",
                data: {
                    url: url
                },
                async: false,
            }).responseText;
            // Returned as a string so need to convert to array
            // The array is returned with single quotes, we need double for JSON.parse
            if (loadURL == 1) {
                console.log("Couldn't scrape page text");
            } else if (loadURL == 2) {
                console.log("Couldn't open model");
            } else if (loadURL == 3) {
                console.log("Model couldn't predict output");
            } else if (loadURL == 4) {
                console.log("Couldn't find topics");
            } else if (loadURL == 5) {
                console.log("Couldn't preprocess website text");
            } else if (loadURL == 6) {
                console.log("Confidence score not produced");
            } else if (loadURL == 7) {
                console.log("Final array not produced");
            }
            newStr = loadURL.replaceAll("'", '"');
            var output = JSON.parse(newStr);
            var classification = output[0]
            var confidence = output[1]
            var topics = output[2]
            // Console the following if any of our model outputs are missing
            if (!classification || !confidence || !topics) {
                console.log("Error model did not return all data");
            }
            // Send the formatting of the classification, topics and confidence score to a function
            formatClassification(classification)
            formatConfidence(confidence)
            formatTopics(topics)
            loading()
        } catch (err) {
            // If it fails then print error message
            str = "<h1>System Error</h1><p>Please contact System Administrator</p> " + err.message
            document.getElementById("loading").innerHTML = str;
        }

    }
};