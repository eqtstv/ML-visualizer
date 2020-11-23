class ProjectsList extends HTMLElement {
  constructor() {
    super();
    const data = document.getElementById("data-container");
    const exampleAttr = data.getAttribute("data-");
    const toParse = exampleAttr.replace(/'/g, '"');
    const projects = JSON.parse(toParse);
    const container = document.getElementsByClassName("projects-container")[0];

    for (var i = 0; i < Object.keys(projects).length; i++) {
      var cardDiv = document.createElement("div");
      cardDiv.setAttribute("class", "project-card");
      cardDiv.setAttribute("id", "project" + i);

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
        console.log(e.target.children[0].innerHTML);
      };
      projectTitle.onclick = (e) => {
        e.stopPropagation();
        console.log(e.target.parentNode.children[0].innerHTML);
      };
      projectDescription.onclick = (e) => {
        e.stopPropagation();
        console.log(e.target.parentNode.children[0].innerHTML);
      };

      cardDiv.appendChild(projectTitle);
      cardDiv.appendChild(projectDescription);
      container.appendChild(cardDiv);
    }
  }
}
customElements.define("projects-list", ProjectsList);
