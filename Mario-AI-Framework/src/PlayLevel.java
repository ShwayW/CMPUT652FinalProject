import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import engine.core.MarioGame;
import engine.core.MarioResult;

public class PlayLevel {
    public static void printResults(MarioResult result) {
        System.out.println("****************************************************************");
        System.out.println("Game Status: " + result.getGameStatus().toString() +
                " Percentage Completion: " + result.getCompletionPercentage());
        System.out.println("Lives: " + result.getCurrentLives() + " Coins: " + result.getCurrentCoins() +
                " Remaining Time: " + (int) Math.ceil(result.getRemainingTime() / 1000f));
        System.out.println("Mario State: " + result.getMarioMode() +
                " (Mushrooms: " + result.getNumCollectedMushrooms() + " Fire Flowers: " + result.getNumCollectedFireflower() + ")");
        System.out.println("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp() +
                " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() +
                " Falls: " + result.getKillsByFall() + ")");
        System.out.println("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps() +
                " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime());
        System.out.println("****************************************************************");
    }

    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
        }
        return content;
    }

    public static void main(String[] args) {
        
        //printResults(game.playGame(getLevel("./levels/original/lvl-1.txt"), 200, 0));
        //printResults(game.playGame(getLevel("../../output/output_01path.txt"), 200, 0));
        // would also need to adjust the positionData.txt
        for(int i=1; i <=20; i++){
            MarioGame game = new MarioGame();
            String filePath = String.format("../../output_transformer/completionist/output_%dpath.txt", i);
            System.out.println(filePath);
            try{
                printResults(game.runGame(new agents.robinBaumgarten.Agent(),getLevel(filePath), 60, 0));
            }
            catch(Exception e){
                System.out.println("Unplayable");
            }
        }
        
        //printResults(game.runGame(new agents.robinBaumgarten.Agent(), getLevel("../../output/completionist/output_01path.txt"), 200, 0, true));
        //printResults(game.runGame(new agents.robinBaumgarten.Agent(), getLevel("./levels/original/lvl-2.txt"), 20, 0, true));
        //printResults(game.runGame(new agents.robinBaumgarten.Agent(), getLevel("./levels/original/lvl-3.txt"), 20, 0, true));
    }
}
