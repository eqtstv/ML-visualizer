class ProjectsList extends HTMLElement {
  constructor() {
    super();
    let sendCurrentProject = async () => {
      const rawResponse = await fetch("http://127.0.0.1:5050/current_project", {
        method: "PUT",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          current_project: localStorage.getItem("current-project"),
        }),
      });
    };

    const data = document.getElementById("data-container");
    const exampleAttr = data.getAttribute("data-");
    const toParse = exampleAttr.replace(/'/g, '"');
    const projects = JSON.parse(toParse);
    const container = document.getElementsByClassName("projects-container")[0];

    for (var i = 0; i < Object.keys(projects).length; i++) {
      var labelWrap = document.createElement("label");
      var cardDiv = document.createElement("div");
      var radioInput = document.createElement("input");

      cardDiv.setAttribute("class", "project-card");

      radioInput.setAttribute("type", "radio");
      radioInput.setAttribute("name", "test");

      var projectTitle = document.createElement("p");
      projectTitle.setAttribute("class", "project-title");
      var projectDescription = document.createElement("p");
      projectDescription.setAttribute(
        "class",
        "project-description has-text-left"
      );
      projectTitle.innerHTML = projects["project" + i]["name"];
      projectDescription.innerHTML = projects["project" + i]["description"];

      cardDiv.onclick = (e) => {
        e.stopPropagation();
        localStorage.setItem("current-project", e.target.children[0].innerHTML);
        sendCurrentProject();
      };
      projectTitle.onclick = (e) => {
        e.stopPropagation();
        localStorage.setItem(
          "current-project",
          e.target.parentNode.children[0].innerHTML
        );
        sendCurrentProject();
      };
      projectDescription.onclick = (e) => {
        e.stopPropagation();
        localStorage.setItem(
          "current-project",
          e.target.parentNode.children[0].innerHTML
        );
        sendCurrentProject();
      };

      cardDiv.appendChild(projectTitle);
      cardDiv.appendChild(projectDescription);
      labelWrap.appendChild(radioInput);
      labelWrap.appendChild(cardDiv);
      container.appendChild(labelWrap);
    }
  }
}
customElements.define("projects-list", ProjectsList);
