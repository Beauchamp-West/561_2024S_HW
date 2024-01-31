import java.util.*;
import java.io.*;
// import java.nio.file.Files;
// import java.nio.file.Paths;
// import java.nio.file.StandardOpenOption;

/*
 * A graph representation of all safe locations and path segments in a 
 * 3D path planning problem. This class also includes three searching algorithms 
 * for solving the problem in three different scenarios.
 */
class LocationGraph {
    private String algoName;
    private int uphillEnergy;
    private GraphNode start;

    public LocationGraph(String inputPath) {
        // initialize the graph.
        createGraphFromInput(inputPath);
        // System.out.println(start);
        // System.out.println(goal);
    }

    public void solve() {
        List<String> shortestPath;
        if (algoName.equals("BFS")) {
            shortestPath = bfs();
        } else if (algoName.equals("UCS")) {
            shortestPath = ucs();
        } else {
            shortestPath = aPrime();
        }
        outputPath(shortestPath);
    }

    /*
     * Evaluate the input file to create a graph of safe locations and path segments.
     * Locations are represented by nodes and the path segments between locations are
     * represented by edges.
     */
    private void createGraphFromInput(String inputPath) {
        try {
            File inputFile = new File(inputPath);
            Scanner scanner = new Scanner(inputFile);

            this.algoName = scanner.nextLine();
            this.uphillEnergy = Integer.parseInt(scanner.nextLine());

            int numLocs = Integer.parseInt(scanner.nextLine());
            // store all locations as nodes in a name-to-node map.
            Map<String, GraphNode> graphNodeMap = new HashMap<>();
            Map<String, Set<Integer>> neighborMomentum = new HashMap<>();
            for (int i = 0; i < numLocs; i++) {
                String[] elements = scanner.nextLine().split(" ");
                assert elements.length == 4;
                String name = elements[0];
                int x = Integer.parseInt(elements[1]);
                int y = Integer.parseInt(elements[2]); 
                int z = Integer.parseInt(elements[3]);
                GraphNode node = new GraphNode(name, x, y, z);
                graphNodeMap.put(name, node);
                neighborMomentum.put(name, new HashSet<>());
                if (name.equals("start")) start = node;
            }

            int numPathes = Integer.parseInt(scanner.nextLine());
            // add the endpoints of each path segment to the neighborhood of each other.
            for (int j = 0; j < numPathes; j++) {
                String[] nodePair = scanner.nextLine().split(" ");
                assert nodePair.length == 2;
                GraphNode node0 = graphNodeMap.get(nodePair[0]), node1 = graphNodeMap.get(nodePair[1]);
                node0.neighbors.add(node1);
                node1.neighbors.add(node0);
                int z0 = node0.z, z1 = node1.z;
                neighborMomentum.get(nodePair[0]).add(Math.max(0, z1-z0-uphillEnergy));
                neighborMomentum.get(nodePair[1]).add(Math.max(0, z0-z1-uphillEnergy));
            }
            scanner.close();

            for (String nodeName : neighborMomentum.keySet()) {
                neighborMomentum.get(nodeName).add(Integer.MAX_VALUE);
                graphNodeMap.get(nodeName).sortedMomentum = neighborMomentum.get(nodeName).toArray(Integer[]::new);
                Arrays.sort(graphNodeMap.get(nodeName).sortedMomentum);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void outputPath(List<String> path) {
        try {
            FileWriter writer = new FileWriter("./output.txt");
            String res;
            if (path.isEmpty()) {
                res = "FAIL";
            } else {
                res = String.join(" ", path);
            }
            writer.write(res + "\n");
            writer.close();

            writer = new FileWriter("./pathlen.txt");
            writer.write(String.valueOf(path.size()-1));
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private List<String> bfs() {
        // location -> momentum-based partition
        Map<String, Set<Integer>> visited = new HashMap<>();
        Set<Integer> set = new HashSet<>();
        set.add(0);
        visited.put(start.name, set);
        Queue<SearchNode> queue = new LinkedList<>();
        SearchNode init = new SearchNode(start);
        queue.offer(init);

        while (!queue.isEmpty()) {
            SearchNode curr = queue.poll();
            GraphNode state = curr.state;
            for (GraphNode neighbor : state.neighbors) {
                // ignore the neighbor without sufficient energy to access
                if (!curr.isValidChild(neighbor, uphillEnergy)) continue;

                int mo = Math.max(0, state.z - neighbor.z);
                SearchNode child = new SearchNode(neighbor, curr, mo);
                if (neighbor.isGoal()) return child.path();

                // ignore the neighbor with the same momentum partition visited before
                Set<Integer> momentums = visited.get(neighbor.name);
                int momentState = neighbor.getMomentumState(mo);
                if (momentums != null && momentums.contains(momentState)) continue;
                else if (momentums == null) {
                    momentums = new HashSet<>();
                    visited.put(neighbor.name, momentums);
                }
                momentums.add(momentState);
                queue.offer(child);
            }
        }

        return new ArrayList<>();
    }

    private List<String> ucs() {
        // name -> {momentumState -> node}
        Map<String, Map<Integer, UCSNode>> open = new HashMap<>();
        Map<String, Map<Integer, UCSNode>> closed = new HashMap<>();
        UCSNode init = new UCSNode(start);
        Map<Integer, UCSNode> map = new HashMap<>();
        map.put(start.getMomentumState(0), init);
        open.put(start.name, map);
        PriorityQueue<UCSNode> pq = new PriorityQueue<>((a, b) -> Double.compare(a.cost, b.cost));
        
        pq.offer(init);

        while (!pq.isEmpty()) {
            UCSNode curr = pq.poll();
            GraphNode state = curr.state;
            if (state.isGoal()) return curr.path();
            int momentState = state.getMomentumState(curr.momentum);
            open.get(state.name).remove(momentState);

            for (GraphNode neighbor : state.neighbors) {
                // ignore the neighbor without sufficient energy to access
                if (!curr.isValidChild(neighbor, uphillEnergy)) continue;

                int mo = Math.max(0, state.z - neighbor.z);
                momentState = neighbor.getMomentumState(mo);
                int sx = state.x, sy = state.y, nx = neighbor.x, ny = neighbor.y;
                double cost = Math.sqrt(Math.pow(sx-nx, 2) + Math.pow(sy-ny, 2));
                UCSNode child = new UCSNode(neighbor, curr, mo, cost);

                Map<Integer, UCSNode> openMap = open.get(neighbor.name);
                if (openMap == null) {
                    openMap = new HashMap<>();
                    open.put(neighbor.name, openMap);
                }
                UCSNode openNode = openMap.get(momentState);
                Map<Integer, UCSNode> closedMap = closed.get(neighbor.name);
                if (closedMap == null) {
                    closedMap = new HashMap<>();
                    closed.put(neighbor.name, closedMap);
                }
                UCSNode closedNode = closedMap.get(momentState);
                
                if (openNode == null && closedNode == null) {
                    pq.offer(child);
                    openMap.put(momentState, child);
                } else if (openNode != null) {
                    if (child.cost < openNode.cost) {
                        pq.remove(openNode);
                        pq.offer(child);
                        openMap.put(momentState, child);
                    }
                } else if (child.cost < closedNode.cost) {
                    closedMap.put(momentState, child);
                    pq.offer(child);
                }
            }
        }

        return new ArrayList<>();
    }

    private List<String> aPrime() {
        List<String> res = new ArrayList<>();
        Queue<GraphNode> queue = new LinkedList<>();
        queue.offer(start);


        return res;
    }  
}

class GraphNode {
    public String name;
    public int x, y, z;
    public Set<GraphNode> neighbors;
    public Integer[] sortedMomentum;

    public GraphNode(String name, int x, int y, int z) {
        this.name = name;
        this.x = x;
        this.y = y;
        this.z = z;
        neighbors = new HashSet<GraphNode>();
    }

    public boolean isGoal() {
        return name.equals("goal");
    }

    // discretize the 'continous' momentum based on its accessible neighbors
    public int getMomentumState(int momentum) {
        int l = 0, r = sortedMomentum.length - 2;
        while (l <= r) {
            int m = (l + r) >> 1;
            if (momentum >= sortedMomentum[m+1]) l = m+1;
            else if (momentum < sortedMomentum[m]) r = m-1;
            else return m;
        }

        return -1;
    }

    public String toString() {
        return String.format("name=%s, (x,y,z)=(%d,%d,%d)", name, x, y, z);
    }
}

class SearchNode {
    public GraphNode state;
    public SearchNode parent;
    public int momentum;

    public SearchNode(GraphNode state) {
        this.state = state;
        this.parent = null;
        this.momentum = 0;
    }

    public SearchNode(GraphNode state, SearchNode parent, int momentum) {
        this.state = state;
        this.parent = parent;
        this.momentum = momentum;
    }

    public boolean isValidChild(GraphNode gNode, int energy) {
        return energy + momentum + state.z >= gNode.z;
    }

    public List<String> path() {
        Deque<String> res = new ArrayDeque<>();
        SearchNode prev = this;
        while (prev != null) {
            res.addFirst(prev.state.name);
            prev = prev.parent;
        }

        return new ArrayList<>(res);
    }

    public String toString(int depth) {
        if (parent == null) {
            return String.format("%s, parent=null, mom=%d, depth=%d\n", state.toString(), momentum, depth);
        } else {
            return String.format("%s, parent=%s, mom=%d, depth=%d\n", state.toString(), parent.state.name, momentum, depth);
        }
    }
}

class UCSNode extends SearchNode {
    public double cost;

    public UCSNode(GraphNode state) {
        super(state);
        cost = 0d;
    }

    public UCSNode(GraphNode state, UCSNode parent, int momentum, double cost) {
        super(state, parent, momentum);
        this.cost = parent.cost + cost;
    }
    
}