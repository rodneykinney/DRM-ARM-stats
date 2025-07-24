import random
import webbrowser
import tempfile
import os

class TreeNode:
    def __init__(self, text="", style=None, left=None, right=None):
        self.text = text  # Can be single line or multi-line string
        self.style = style or {}  # Dictionary for text styling options
        self.left = left
        self.right = right

    def get_lines(self):
        """Split text into lines for multi-line display"""
        if isinstance(self.text, str):
            return self.text.split('\n')
        return [str(self.text)]

    def is_leaf(self):
        """Check if a node is a leaf (has no children)"""
        return self.left is None and self.right is None

class BinaryTreeHTMLGenerator:
    def __init__(self, root):
        self.root = root
        self.colors = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
            '#9b59b6', '#1abc9c', '#e67e22', '#34495e',
            '#e91e63', '#607d8b', '#795548', '#ff5722'
        ]
        self.node_counter = 0
        self.min_node_width = 80
        self.min_node_height = 40
        self.padding = 20

    def get_color(self, depth):
        """Get color for a given depth"""
        return self.colors[depth % len(self.colors)]

    def estimate_text_size(self, text, font_size):
        """Estimate text dimensions based on character count and font size"""
        lines = text.split('\n') if isinstance(text, str) else [str(text)]
        
        # Rough estimates: 0.6 * font_size per character width, 1.3 * font_size per line height
        char_width = font_size * 0.6
        line_height = font_size * 1.3
        
        max_line_length = max(len(line) for line in lines) if lines else 0
        num_lines = len(lines)
        
        width = max_line_length * char_width + self.padding * 2
        height = num_lines * line_height + self.padding * 2
        
        # Ensure minimum dimensions
        width = max(width, self.min_node_width)
        height = max(height, self.min_node_height)
        
        return width, height

    def calculate_tree_dimensions(self, node, depth=0):
        """Calculate dimensions for entire tree structure with nested layout"""
        if not node:
            return 0, 0, {}
        
        font_size = node.style.get('fontSize', max(10, 16 - depth * 1.5))
        base_width, base_height = self.estimate_text_size(node.text, font_size)
        
        # Calculate children dimensions
        left_width, left_height, left_dims = self.calculate_tree_dimensions(node.left, depth + 1)
        right_width, right_height, right_dims = self.calculate_tree_dimensions(node.right, depth + 1)
        
        # For nested layout, parent must contain children
        children_width = left_width + right_width
        if node.left and node.right:
            children_width += self.padding  # Space between children
        
        # Parent width must accommodate text and children side by side
        min_width_for_children = children_width + self.padding * 2 if (node.left or node.right) else 0
        node_width = max(base_width, min_width_for_children)
        
        # Parent height must accommodate text and children below it
        children_height = max(left_height, right_height) if (node.left or node.right) else 0
        text_height = base_height
        node_height = text_height + children_height + (self.padding if children_height > 0 else 0)
        
        # Store node dimensions
        node_key = id(node)
        dimensions = {node_key: (node_width, node_height)}
        
        # Merge dimension dictionaries
        dimensions.update(left_dims)
        dimensions.update(right_dims)
        
        return node_width, node_height, dimensions

    def generate_css(self):
        """Generate the CSS styles for the tree visualization"""
        return """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                text-align: center;
            }
            
            .container {
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                display: inline-block;
                min-width: 400px;
            }
            
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 2.5em;
                font-weight: 300;
            }
            
            .tree-container {
                position: relative;
                width: 100%;
                min-height: 400px;
                border: 3px solid #ddd;
                border-radius: 10px;
                overflow: auto;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            }
            
            .tree-node {
                position: absolute;
                border: 2px solid #333;
                border-radius: 8px;
                display: flex;
                flex-direction: column;
                box-sizing: border-box;
            }
            
            .node-content {
                padding: 10px;
                text-align: center;
                font-weight: bold;
                color: white;
                // text-shadow: 1px 1px 1px rgba(0,0,0,0.6);
                line-height: 1.3;
                overflow: hidden;
                word-wrap: break-word;
                position: relative;
                z-index: 1;
            }
            
            .leaf-node .node-content {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
            }
            
            .non-leaf-node .node-content {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .info {
                text-align: center;
                color: #666;
                margin-top: 20px;
                font-size: 1.1em;
            }
            
            .tree-stats {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                text-align: center;
                color: #495057;
            }
        </style>
        """

    def generate_node_html(self, node, x, y, node_dimensions, depth=0):
        """Generate HTML for a single node and its children using nested layout"""
        if not node:
            return ""

        self.node_counter += 1
        node_id = f"node_{self.node_counter}"

        # Get node dimensions
        node_key = id(node)
        width, height = node_dimensions[node_key]

        # Get color and style
        bg_color = node.style.get('color', 'white')
        font_size = node.style.get('fontSize', max(10, 16 - depth * 1.5))
        text_color = node.style.get('text-color', 'black')

        # Determine node class
        node_class = 'leaf-node' if node.is_leaf() else 'non-leaf-node'

        # Format text content
        lines = node.get_lines()
        if len(lines) == 1:
            text_content = lines[0]
        else:
            text_content = '<br>'.join(lines)

        # Calculate text area height
        text_height = self.estimate_text_size(node.text, font_size)[1]

        # Generate node HTML with nested children
        html = f"""
        <div id="{node_id}" class="tree-node {node_class}" 
             style="left: {x}px; top: {y}px; width: {width}px; height: {height}px; 
                    background-color: {bg_color};">
            <div class="node-content" style="font-size: {font_size}px; color: {text_color}; height: {text_height}px;">
                {text_content}
            </div>"""

        # Add children inside the parent node (nested layout)
        if node.left or node.right:
            child_container_y = text_height + self.padding // 2
            child_container_height = height - text_height - self.padding // 2
            
            if node.left and node.right:
                # Both children exist - position them side by side
                left_width, left_height = node_dimensions[id(node.left)]
                right_width, right_height = node_dimensions[id(node.right)]
                
                left_x = self.padding // 2
                right_x = left_x + left_width + self.padding // 2
                
                html += self.generate_node_html(node.left, left_x, child_container_y, node_dimensions, depth + 1)
                html += self.generate_node_html(node.right, right_x, child_container_y, node_dimensions, depth + 1)
                
            elif node.left:
                # Only left child - center it
                left_width, left_height = node_dimensions[id(node.left)]
                left_x = (width - left_width) // 2
                html += self.generate_node_html(node.left, left_x, child_container_y, node_dimensions, depth + 1)
                
            elif node.right:
                # Only right child - center it
                right_width, right_height = node_dimensions[id(node.right)]
                right_x = (width - right_width) // 2
                html += self.generate_node_html(node.right, right_x, child_container_y, node_dimensions, depth + 1)

        html += "</div>"
        return html

    def count_nodes(self, node):
        """Count total nodes in the tree"""
        if not node:
            return 0
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)

    def calculate_depth(self, node):
        """Calculate the maximum depth of the tree"""
        if not node:
            return 0
        return 1 + max(self.calculate_depth(node.left), self.calculate_depth(node.right))

    def generate_html(self, title):
        """Generate complete HTML document"""
        if not self.root:
            return "<html><body><h1>No tree to display</h1></body></html>"

        # Reset counter for each generation
        self.node_counter = 0

        # Calculate tree statistics
        total_nodes = self.count_nodes(self.root)
        max_depth = self.calculate_depth(self.root)

        # Calculate dynamic dimensions
        tree_width, tree_height, node_dimensions = self.calculate_tree_dimensions(self.root)
        
        # Add padding around the tree
        container_width = tree_width + self.padding * 2
        container_height = tree_height + self.padding * 2

        # Position root node in center
        root_width, root_height = node_dimensions[id(self.root)]
        root_x = (container_width - root_width) // 2
        root_y = self.padding

        # Generate tree HTML
        tree_html = self.generate_node_html(self.root, root_x, root_y, node_dimensions)

        # Generate complete HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            {self.generate_css()}
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                
                <div class="tree-stats">
                  Probability of sub-7 DR, given corner/edge configuration counts
                </div>
                
                <div class="tree-container" style="width: {container_width}px; height: {container_height}px;">
                    {tree_html}
                </div>
                
                <div class="info">
                    <ul>
                      <li><b>n_pairs</b>: Number of top pairs
                      <li><b>n_fake_pairs</b>: Number of top fake pairs
                      <li><b>n_side_pairs</b>: Number of side pairs
                      <li><b>corner_orbit_split</b>: Number of bad corners in the HTR orbit with the fewest bad corners
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def save_and_open(self, title, filename=None):
        """Generate HTML and open it in the default browser"""
        html_content = self.generate_html(title)

        if filename is None:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
            filename = temp_file.name
            temp_file.write(html_content)
            temp_file.close()
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)

        # Open in browser
        webbrowser.open(f'file://{os.path.abspath(filename)}')
        print(f"Tree visualization saved to: {filename}")
        return filename

def create_unbalanced_tree():
    """Create an unbalanced binary tree of depth 5 with styled text"""
    root = TreeNode("Root Node", {"fontSize": 18})

    root.left = TreeNode("Left Branch", {"fontSize": 16})
    root.right = TreeNode("Right Branch", {"fontSize": 16})

    root.left.left = TreeNode("Deep\nLeft\nPath", {"fontSize": 14, "color": "#ffffff"})
    root.right.right = TreeNode("Right\nChild", {"fontSize": 14})

    root.left.left.left = TreeNode("Very\nDeep\nLeft\nNode", {"fontSize": 12, "color": "#ffffff"})
    root.right.right.left = TreeNode("Right\nSub\nTree", {"fontSize": 12})
    root.right.right.right = TreeNode("Another\nRight\nBranch", {"fontSize": 12})

    root.left.left.left.right = TreeNode("Maximum\nDepth\nLeft\nSide\nNode", {"fontSize": 11, "color": "#ffffff"})
    root.right.right.right.left = TreeNode("Deep\nRight\nLeaf\nNode", {"fontSize": 11})

    root.right.right.right.left.right = TreeNode("Deepest\nNode\nLevel\nFive\nLeaf", {"fontSize": 10, "color": "#ffffff"})

    return root

def create_sample_tree():
    """Create a sample binary tree with styled text"""
    root = TreeNode("Main Hub", {"fontSize": 18})
    root.left = TreeNode("Left\nBranch\nData", {"fontSize": 14})
    root.right = TreeNode("Right\nSide\nInfo", {"fontSize": 14})
    root.left.left = TreeNode("Sub\nLeft\nDetail\nHere", {"fontSize": 12})
    root.left.right = TreeNode("More\nLeft\nContent", {"fontSize": 12})
    root.right.left = TreeNode("Right\nSub\nItems", {"fontSize": 12})
    root.right.right = TreeNode("Final\nRight\nData", {"fontSize": 12})
    return root

def create_random_tree(max_depth=4, fill_probability=0.6):
    """Create a random binary tree with styled text"""
    words = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
             "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi"]

    def build_tree(depth=0):
        if depth >= max_depth or random.random() > fill_probability:
            return None

        if depth == 0:
            text = random.choice(words)
            style = {"fontSize": 18}
        else:
            num_lines = min(depth + 1, 5)
            text = "\n".join(random.choices(words, k=num_lines))
            style = {"fontSize": max(10, 18 - depth * 2), "color": "#ffffff"}

        node = TreeNode(text, style)
        node.left = build_tree(depth + 1)
        node.right = build_tree(depth + 1)
        return node

    return build_tree()

def create_programming_concepts_tree():
    """Create a tree showing programming concepts"""
    root = TreeNode("Programming", {"fontSize": 20})

    root.left = TreeNode("Data\nStructures", {"fontSize": 16})
    root.right = TreeNode("Algorithms", {"fontSize": 16})

    root.left.left = TreeNode("Linear\nStructures\nArray\nList", {"fontSize": 12})
    root.left.right = TreeNode("Tree\nStructures\nBinary\nAVL", {"fontSize": 12})

    root.right.left = TreeNode("Sorting\nBubble\nQuick\nMerge", {"fontSize": 12})
    root.right.right = TreeNode("Search\nLinear\nBinary\nDFS", {"fontSize": 12})

    root.left.left.left = TreeNode("Stack\nLIFO\nPush\nPop\nTop", {"fontSize": 10})
    root.left.left.right = TreeNode("Queue\nFIFO\nEnqueue\nDequeue\nFront", {"fontSize": 10})

    return root

def main():
    """Main function to demonstrate the tree visualizations"""
    print("Binary Tree HTML Generator")
    print("=" * 50)

    # Example 1: Unbalanced tree of depth 5
    print("1. Creating unbalanced tree of depth 5...")
    unbalanced_tree = create_unbalanced_tree()
    visualizer1 = BinaryTreeHTMLGenerator(unbalanced_tree)
    file1 = visualizer1.save_and_open(title="Unbalanced Binary Tree (Depth 5)")

    input("Press Enter to continue to the next example...")

    # Example 2: Sample tree with styled text
    print("2. Creating sample tree with styled text...")
    sample_tree = create_sample_tree()
    visualizer2 = BinaryTreeHTMLGenerator(sample_tree)
    file2 = visualizer2.save_and_open(title="Sample Tree with Multi-line Text")

    input("Press Enter to continue to the next example...")

    # Example 3: Random tree
    print("3. Creating random tree...")
    random_tree = create_random_tree(max_depth=4, fill_probability=0.7)
    visualizer3 = BinaryTreeHTMLGenerator(random_tree)
    file3 = visualizer3.save_and_open(title="Random Binary Tree")

    input("Press Enter to continue to the final example...")

    # Example 4: Programming concepts tree
    print("4. Creating programming concepts tree...")
    concepts_tree = create_programming_concepts_tree()
    visualizer4 = BinaryTreeHTMLGenerator(concepts_tree)
    file4 = visualizer4.save_and_open(title="Programming Concepts Tree")

    print("\nAll visualizations have been generated and opened in your browser!")
    print(f"Temporary files created:")
    for i, file in enumerate([file1, file2, file3, file4], 1):
        print(f"  {i}. {file}")

if __name__ == "__main__":
    main()