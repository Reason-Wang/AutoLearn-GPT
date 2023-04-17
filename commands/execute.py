from search import google_official_search
from browse import browse_website


def execute_command(command_name, question, model):
    if command_name.lower() == "search":
        search_links = google_official_search(question, num_results=8)
        web_contents = []
        for link in search_links:
            web_content = browse_website(link, model)
            web_contents.append(web_content)

        return web_contents

