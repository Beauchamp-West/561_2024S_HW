public class calibrate {
    public static void main(String[] args) {
        StringBuilder sb = new StringBuilder();

        for (int level = 7; level >= 1; level--) {
            long start = System.currentTimeMillis();

            Game game = new Game("input.txt");
            game.solveTimer(level);

            long end = System.currentTimeMillis();
            double t = (end - start) / 1000d;
            sb.append(t).append("\n");
        }

        Game.output(sb.toString() , "calibration.txt");
    }
}
