public class calibrate {
    public static void main(String[] args) {
        StringBuilder sb = new StringBuilder();

        for (int level = 7; level >= 1; level--) {
            sb.append("max level: ").append(level).append(", ");
            long start = System.currentTimeMillis();

            Game game = new Game("input.txt");
            game.solveTimer(level);

            long end = System.currentTimeMillis();
            double t = (end - start) / 1000d;
            double tPerNode = t / game.numExpandedNodes;
            sb.append("expanded node number: ").append(game.numExpandedNodes).append(", ");
            sb.append("total time: ").append(t).append("s, ");
            sb.append("time per node: ").append(tPerNode).append("s\n");
            sb.append(t).append("\n");
        }

        Game.output(sb.toString() , "calibration.txt");
    }
}
