package com.duckprog.makhos;

import java.io.IOException;
import java.io.InputStream;
import java.util.Vector;

import org.andengine.engine.camera.Camera;
import org.andengine.engine.options.EngineOptions;
import org.andengine.engine.options.ScreenOrientation;
import org.andengine.engine.options.resolutionpolicy.RatioResolutionPolicy;
import org.andengine.entity.modifier.MoveModifier;
import org.andengine.entity.scene.IOnSceneTouchListener;
import org.andengine.entity.scene.Scene;
import org.andengine.entity.scene.background.Background;
import org.andengine.entity.scene.menu.MenuScene;
import org.andengine.entity.scene.menu.MenuScene.IOnMenuItemClickListener;
import org.andengine.entity.scene.menu.item.IMenuItem;
import org.andengine.entity.scene.menu.item.SpriteMenuItem;
import org.andengine.entity.sprite.Sprite;
import org.andengine.entity.util.FPSLogger;
import org.andengine.input.touch.TouchEvent;
import org.andengine.opengl.texture.ITexture;
import org.andengine.opengl.texture.TextureOptions;
import org.andengine.opengl.texture.atlas.bitmap.BitmapTextureAtlas;
import org.andengine.opengl.texture.atlas.bitmap.BitmapTextureAtlasTextureRegionFactory;
import org.andengine.opengl.texture.bitmap.BitmapTexture;
import org.andengine.opengl.texture.region.ITextureRegion;
import org.andengine.opengl.texture.region.TextureRegionFactory;
import org.andengine.ui.activity.LayoutGameActivity;
import org.andengine.util.adt.io.in.IInputStreamOpener;

//import com.ahddtejxswazwdqre.AdController;
//import com.yjvfxhmgkxadkxmw.AdController;
//18/08/2014
import com.bheugyiqywscgscae.AdController;

import android.content.SharedPreferences;
import android.media.MediaPlayer;
import android.opengl.GLES20;
import android.os.AsyncTask;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.util.DisplayMetrics;
import android.view.KeyEvent;

import com.apptracker.android.listener.AppModuleListener;
import com.apptracker.android.track.AppTracker;

public class MainActivity  extends LayoutGameActivity implements
IOnSceneTouchListener, IOnMenuItemClickListener  {
	
	public static boolean threadEnd = true;
	static Vector<int[][]> boardList=new Vector<int[][]>();
	static Vector<int[]> captureList=new Vector<int[]>();
	int[] lastCap = new int[16];
	
	final static int white = 1;
	final static int black = 2;
	final static int whiteKing = 3;
	final static int blackKing = 4;
	final static int empty = 0;
	private static final int MENU_NEW = 0;
	private static final int MENU_MAIN = MENU_NEW+1;
	boolean selectPiece = false;
	boolean selectCheck = false;
	boolean incomplete = false;
	boolean startGame = false;
	
	Think2 thinkThread;
	Engine engine;
	
	int checkMove=0;
	public static int countDraw = 0;
	
	int whiteClear = 0;
	int blackClear = 0;
	
	boolean alreadyMove = false;
	
	int level_select = 2;
	int piece_select = 0;
	boolean twoPlayer = false;
	boolean sound_enable;
	int piece_opponent = 0;
	
	int toMove = black;
	int startTurn = black;
	static int start_i,start_j,end_i,end_j;
	static int w_start_i,w_start_j,w_end_i,w_end_j;
	
	private ITextureRegion mBackgroundRegion, itr_nWhiteKing, itr_undo, itr_undodis;
	private ITextureRegion itr_nBlack,itr_nWhite,itr_nBlackKing;
	private ITextureRegion itr_nBlackDis,itr_nWhiteDis;
	private ITextureRegion itr_lose , itr_win , itr_draw , itr_end , itr_start , itr_think , itr_humanwin, itr_back;
	private ITextureRegion itr_appStore;
	
	static Piece[] wnPiece = new Piece[8];
	static Piece[] bnPiece = new Piece[8];
	
	static Piece[][] nPiece = new Piece[8][8];
	int activeI = 0 , activeJ =0;
	int[][] matrix = new int[8][8];
	
	static Piece nBlackSelect;
	static Piece nBlackDisable;
	static Piece nBlackKingSelect;
	static Piece nBlackKingDisable;
	
	private static int CAMERA_WIDTH = 840; 
	private static int CAMERA_HEIGHT = 1200;
	static int leftWidth = 20;
	static int topHeight = 260;	
	
	Scene mCurrentScene;
	private Sprite bStart , end , lose , win , draw , think;
	private Sprite bTurn , bTurnDis , wTurn , wTurnDis , humanwin , pieceWin, sBack,sUndo,sUndoDis;
	private Sprite sAppStore;
	
	private MediaPlayer mMove,mWin,mLose,mEat,mDisable,mMenu;
	
	//private AdView adView;
	
	private BitmapTextureAtlas mMenuTexture;
	protected ITextureRegion mMenuNewTextureRegion;
	protected ITextureRegion mMenuMenuTextureRegion;
	
	protected MenuScene mMenuScene;
	
	protected Camera mCamera;
	
	private AdController ad;
	private AdController adwall;
	private String sectionid;
	private int pointTop;
	@Override
	protected void onDestroy(){
		ad.destroyAd();
		adwall.destroyAd();
		super.onDestroy();
	}
	
	@Override
	protected void onCreate(Bundle pSavedInstanceState) {
		super.onCreate(pSavedInstanceState);
		
		
		if(pSavedInstanceState == null) {
            // Initialize Leadbolt SDK with your api key
			AppTracker.startSession(getApplicationContext(),"lfli7sPXV7MIL1u9azv8dzjqc7ENU6ty");
        }
        // cache Leadbolt Ad without showing it
        AppTracker.loadModuleToCache(getApplicationContext(),"inapp");
		DisplayMetrics dm = new DisplayMetrics();
		  getWindowManager().getDefaultDisplay().getMetrics(dm);
		  int screenWidth = dm.widthPixels;
		  pointTop = dm.widthPixels/2-420;
		  /*sectionid = "383773578";
		  if(screenWidth >= 720) {
		  sectionid = "308799023";
		  }
		  else if(screenWidth >= 640) {
		  sectionid = "245361661";
		  }
		  else if(screenWidth >= 468) {
		  sectionid = "909494006";
		  }*/
		  sectionid = "134531518";
		  ad = new AdController(this, sectionid);
		  /*if (pointTop>0){
			  ad.setAdditionalDockingMargin(pointTop);
		  }
		  else{
			  ad.setAdditionalDockingMargin(0);
		  }*/
		  ad.setAdditionalDockingMargin(0);
		  adwall = new AdController(this, "441825600");
		  ad.loadAd();
	}

	@Override
	public void onResume()
	{
		super.onResume();
		ad.destroyAd();
		/*ad = new AdController(this, sectionid);
		if (pointTop>0){
			ad.setAdditionalDockingMargin(pointTop);
		}
		else{
			ad.setAdditionalDockingMargin(0);
		}*/
		ad.setAdditionalDockingMargin(0);
		ad.loadAd();//*/
		
		adwall.destroyAd();
		adwall = new AdController(this, "441825600");
	}
	
	@Override
	public void onBackPressed() {
		// TODO Auto-generated method stub
		super.onBackPressed();
		
		//System.out.println("step back process");
		if (engine != null){
			//System.out.println("step back process engine");
			if (!engine.stoprun){
				engine.stoprun = true;
				pause(1000);
				/*while (!threadEnd){
				
				}*/
			}
			
			engine = null;
		}

		//System.out.println("step back process exit");

		threadEnd=false;
		
		if (thinkThread!=null){
			thinkThread = null;
		}
	}

	@Override
	protected void onSetContentView() {
		// TODO Auto-generated method stub
		super.onSetContentView();
		        
	}	
	
	@Override
	public EngineOptions onCreateEngineOptions() {
		// TODO Auto-generated method stub
		//final Camera camera = new Camera(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
		this.mCamera= new Camera(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
		return new EngineOptions(false, ScreenOrientation.PORTRAIT_FIXED,
				new RatioResolutionPolicy(CAMERA_WIDTH, CAMERA_HEIGHT), mCamera);
	}

	@Override
	public void onCreateResources(
			OnCreateResourcesCallback pOnCreateResourcesCallback)
			throws Exception {
		// TODO Auto-generated method stub
		try {
			
			ITexture backgroundTexture = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/board1.jpg");
						}
					});
			
			SharedPreferences app_preferences = 
			         PreferenceManager.getDefaultSharedPreferences(this);
			        
			        level_select = app_preferences.getInt("level_select", 2);
			        
			        piece_select = app_preferences.getInt("piece_select", 0);
			        
			        twoPlayer = app_preferences.getBoolean("player2", false);

			        sound_enable = app_preferences.getBoolean("sound_enable", false);
			
			piece_opponent = (int)(Math.random()*4);
			//System.out.println("step p : "+piece_select+ " o : "+piece_opponent);
			if (piece_opponent == piece_select){
				if (piece_opponent == 0){
					piece_opponent++;
				}
				else if (piece_opponent == 3){
					piece_opponent--;
				}
				else{
					piece_opponent++;
				}
			}
			
			ITexture nWhite = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							if (piece_opponent == 0){
								return getAssets().open("gfx/piece2.png");
							}
							else if (piece_opponent == 1){
								return getAssets().open("gfx/piece1.png");
							}
							else if (piece_opponent == 2){
								return getAssets().open("gfx/piece3.png");
							}
							else{
								return getAssets().open("gfx/piece4.png");
							}
						}
					});
			
			ITexture nWhiteDis = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							if (piece_opponent == 0){
								return getAssets().open("gfx/piece2dis.png");
							}
							else if (piece_opponent == 1){
								return getAssets().open("gfx/piece1dis.png");
							}
							else if (piece_opponent == 2){
								return getAssets().open("gfx/piece3dis.png");
							}
							else{
								return getAssets().open("gfx/piece4dis.png");
							}
						}
					});
			ITexture nWhiteKing = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							if (piece_opponent == 0){
								return getAssets().open("gfx/pieceKing2.png");
							}
							else if (piece_opponent == 1){
								return getAssets().open("gfx/pieceKing1.png");
							}
							else if (piece_opponent == 2){
								return getAssets().open("gfx/pieceKing3.png");
							}
							else{
								return getAssets().open("gfx/pieceKing4.png");
							}
						}
					});
			

			ITexture nBlack = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							if (piece_select == 0){
								return getAssets().open("gfx/piece2.png");
							}
							else if (piece_select == 1){
								return getAssets().open("gfx/piece1.png");
							}
							else if (piece_select == 2){
								return getAssets().open("gfx/piece3.png");
							}
							else{
								return getAssets().open("gfx/piece4.png");
							}
						}
					});
			
			ITexture nBlackDis = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							if (piece_select == 0){
								return getAssets().open("gfx/piece2dis.png");
							}
							else if (piece_select == 1){
								return getAssets().open("gfx/piece1dis.png");
							}
							else if (piece_select == 2){
								return getAssets().open("gfx/piece3dis.png");
							}
							else{
								return getAssets().open("gfx/piece4dis.png");
							}
						}
					});

			ITexture nBlackKing = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							if (piece_select == 0){
								return getAssets().open("gfx/pieceKing2.png");
							}
							else if (piece_select == 1){
								return getAssets().open("gfx/pieceKing1.png");
							}
							else if (piece_select == 2){
								return getAssets().open("gfx/pieceKing3.png");
							}
							else{
								return getAssets().open("gfx/pieceKing4.png");
							}
						}
					});

			ITexture nEmpty = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/empty.png");
						}
					});	
			ITexture lose = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/lose.png");
						}
					});	
			ITexture win = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/win.png");
						}
					});	
			ITexture humanwin = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/humanwin.png");
						}
					});	
			ITexture draw = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/draw.png");
						}
					});	
			ITexture end = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {

						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/end.jpg");
						}
					});	
			ITexture bStartTexture = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/playagain.png");
						}
	
					});
			ITexture iThink = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/think.png");
						}
	
					});
			
			ITexture iBack = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/backtomenu.png");
						}
	
					});
			
			ITexture iundo = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/undomove.png");
						}
	
					});
			
			ITexture iundodis = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/undomovedis.png");
						}
	
					});
			
			ITexture iAppStore = new BitmapTexture(getTextureManager(),
					new IInputStreamOpener() {
	
						@Override
						public InputStream open() throws IOException {
							// TODO Auto-generated method stub
							return getAssets().open("gfx/appstore.png");
						}
	
					});
			
			backgroundTexture.load();
			nWhite.load();
			nWhiteDis.load();
			nWhiteKing.load();
			nBlack.load();
			nBlackDis.load();
			nBlackKing.load();
			nEmpty.load();
			lose.load();
			win.load();
			end.load();
			draw.load();
			bStartTexture.load();
			humanwin.load();
			iThink.load();
			iBack.load();
			iundo.load();
			iundodis.load();
			iAppStore.load();

			this.itr_nWhite = TextureRegionFactory.extractFromTexture(nWhite);
			this.itr_nWhiteDis = TextureRegionFactory.extractFromTexture(nWhiteDis);
			this.itr_nWhiteKing = TextureRegionFactory.extractFromTexture(nWhiteKing);
			this.itr_nBlack = TextureRegionFactory.extractFromTexture(nBlack);	
			this.itr_nBlackDis = TextureRegionFactory.extractFromTexture(nBlackDis);
			this.itr_nBlackKing = TextureRegionFactory.extractFromTexture(nBlackKing);
			this.itr_draw = TextureRegionFactory.extractFromTexture(draw);
			this.itr_win = TextureRegionFactory.extractFromTexture(win);
			this.itr_lose = TextureRegionFactory.extractFromTexture(lose);
			this.itr_end = TextureRegionFactory.extractFromTexture(end);
			this.itr_start = TextureRegionFactory.extractFromTexture(bStartTexture);
			this.itr_think = TextureRegionFactory.extractFromTexture(iThink);
			this.itr_humanwin = TextureRegionFactory.extractFromTexture(humanwin);
			this.itr_back = TextureRegionFactory.extractFromTexture(iBack);
			this.itr_undo = TextureRegionFactory.extractFromTexture(iundo);
			this.itr_undodis = TextureRegionFactory.extractFromTexture(iundodis);
			this.itr_appStore = TextureRegionFactory.extractFromTexture(iAppStore);
			this.mBackgroundRegion = TextureRegionFactory
					.extractFromTexture(backgroundTexture);
			
			} catch (Exception e) {
				e.printStackTrace();
			}
		
		BitmapTextureAtlasTextureRegionFactory.setAssetBasePath("gfx/");
		this.mMenuTexture = new BitmapTextureAtlas(this.getTextureManager(), 1200, 400, TextureOptions.BILINEAR);
		this.mMenuNewTextureRegion = BitmapTextureAtlasTextureRegionFactory.createFromAsset(this.mMenuTexture, this, "newgame.png", 0, 0);
		this.mMenuMenuTextureRegion = BitmapTextureAtlasTextureRegionFactory.createFromAsset(this.mMenuTexture, this, "backtomenu.png", 0, 200);
		this.mMenuTexture.load();
		
		mMove = MediaPlayer.create(getBaseContext(), R.raw.move2);
		mMove.setLooping(false);
		
		mWin = MediaPlayer.create(getBaseContext(), R.raw.win);
		mWin.setLooping(false);
		
		mLose = MediaPlayer.create(getBaseContext(), R.raw.lose);
		mLose.setLooping(false);
		
		mEat = MediaPlayer.create(getBaseContext(), R.raw.eat);
		mEat.setLooping(false);
		
		mMenu = MediaPlayer.create(getBaseContext(), R.raw.menu);
		mMenu.setLooping(false);
		
		mDisable = MediaPlayer.create(getBaseContext(), R.raw.disable);
		mDisable.setLooping(false);
		
		pOnCreateResourcesCallback.onCreateResourcesFinished();		
	}

	@Override
	public void onCreateScene(OnCreateSceneCallback pOnCreateSceneCallback)
			throws Exception {
		// TODO Auto-generated method stub
		//System.out.println("step t : "+twoPlayer);

		this.mEngine.registerUpdateHandler(new FPSLogger());
		this.createMenuScene();
		boardList.removeAllElements();
		captureList.removeAllElements();
		
		startGame = true;

		mCurrentScene = new Scene();

		mCurrentScene.setBackground(new Background(0.09804f, 0.7274f, 0.8f));
		Sprite bg = new Sprite(0, 0, this.mBackgroundRegion,
				getVertexBufferObjectManager());
		
		sUndo = new Sprite(630, 1090, this.itr_undo,
				getVertexBufferObjectManager()){
					public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
							float pTouchAreaLocalX, float pTouchAreaLocalY) {
						
						if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
							undoLastTurn();
						}
						return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
								pTouchAreaLocalY);
					}
				};
		sUndoDis = new Sprite(630, 1090, this.itr_undodis,
				getVertexBufferObjectManager());
		
		if (!twoPlayer){
			sUndoDis.setVisible(true);
			sUndo.setVisible(false);
		}
		else{
			sUndoDis.setVisible(false);
			sUndo.setVisible(false);
		}
		
		bTurnDis = new Sprite(525, 1080, this.itr_nBlackDis,
				getVertexBufferObjectManager());
		bTurnDis.setScale((float) 0.9);
		bTurn = new Sprite(525, 1080, this.itr_nBlack,
				getVertexBufferObjectManager());
		bTurn.setScale((float) 0.9);
		bTurn.setVisible(false);
		
		wTurnDis = new Sprite(525, 135, this.itr_nWhiteDis,
				getVertexBufferObjectManager());
		wTurnDis.setScale((float) 0.9);
		wTurn = new Sprite(525, 135, this.itr_nWhite,
				getVertexBufferObjectManager());
		wTurn.setScale((float) 0.9);
		wTurn.setVisible(false);
		
		if (toMove == black){
			bTurn.setVisible(true);
			wTurn.setVisible(false);
		}
		else{
			bTurn.setVisible(false);
			wTurn.setVisible(true);
		}
		
		mCurrentScene.attachChild(bg);
		mCurrentScene.attachChild(sUndo);
		mCurrentScene.attachChild(sUndoDis);
		mCurrentScene.attachChild(bTurnDis);
		mCurrentScene.attachChild(bTurn);
		mCurrentScene.attachChild(wTurnDis);
		mCurrentScene.attachChild(wTurn);
		
		mCurrentScene.registerTouchArea(sUndo);
		/*initMatrix();
		initPiecePosition();*/
		
		initGame();
		
		think = new Sprite(390, 610, this.itr_think,
				getVertexBufferObjectManager());
		think.setVisible(false);
		mCurrentScene.attachChild(think);

		mCurrentScene.setOnSceneTouchListener(this);
		mCurrentScene.setTouchAreaBindingOnActionDownEnabled(true);
		
		pOnCreateSceneCallback.onCreateSceneFinished(mCurrentScene);
		/*if (toMove==white){
			checkMove=0;
			bTurn.setVisible(false);
			wTurn.setVisible(true);
			new Think2().start();
		}*/
	}

	@Override
	public void onPopulateScene(Scene pScene,
			OnPopulateSceneCallback pOnPopulateSceneCallback) throws Exception {
		// TODO Auto-generated method stub
		pOnPopulateSceneCallback.onPopulateSceneFinished();
	}
	
	class Think2 extends Thread{

		@Override
		public void run() {
			// TODO Auto-generated method stub
			super.run();
			switchToMove();
			
		}		
		
		/*public void destroy(){
			super.destroy();
		}*/
		
		
	}
	
	class Think extends AsyncTask<String,String,String>{
		
		@Override
		protected void onPostExecute(String result) {
			// TODO Auto-generated method stub
			super.onPostExecute(result);
		}

		@Override
		protected void onPreExecute() {
			// TODO Auto-generated method stub
			super.onPreExecute();
			
		}

		@Override
		protected String doInBackground(String... params) {
			// TODO Auto-generated method stub
			switchToMove();
			return "";
		}
		
	}
	@Override
	public boolean onSceneTouchEvent(Scene pScene, TouchEvent pSceneTouchEvent) {
		// TODO Auto-generated method stub
		
		if (startGame){
			if (bStart!=null){
				mCurrentScene.unregisterTouchArea(bStart);
			}
			if (sBack!=null){
				mCurrentScene.unregisterTouchArea(sBack);
			}
			if (sAppStore!=null){
				mCurrentScene.unregisterTouchArea(sAppStore);
			}
			int i = (int) ((pSceneTouchEvent.getX()-leftWidth) / 100);
			int j = (int) ((pSceneTouchEvent.getY()-topHeight) / 100);
			if (i>=0 && i<8 && j>=0 && j<8){
				
				if (!incomplete && toMove==black && (matrix[i][j] == black || matrix[i][j] == blackKing)){
					//System.out.println("step b 1");
					if(selectCheck){
						//System.out.println("step b 1.1");
						if(!incomplete){
							//System.out.println("step b 1.2");
							if (start_i == i && start_j == j){
								//System.out.println("step b 1.1 same position");
							}
							else{
								selectPiece = true;
								start_i = i;
								start_j = j;
								if(! Move.canCapture(matrix,start_i,start_j) && Move.canCapture(matrix,toMove)){
									//System.out.println("step b 1.3");
									if(matrix[i][j] == 2){
										if (sound_enable){
											mDisable.start();
										}
										disablePieceBlack(i,j);
										return false;
									}
									else{
										if (sound_enable){
											mDisable.start();
										}
										disablePieceBlack(i,j);
										return false;
									}
								}
								else{
									if(matrix[i][j] == 2 || matrix[i][j] == 5){	
										activePieceBlack(i,j);
										return false;
									}
									else{
										activePieceBlack(i,j);
										return false;
									}				
								}
							}
						}
						selectCheck = false;
					}
					else{
						//System.out.println("step b 1.3");
						//disablePieceBlack(i,j);
					}
				} 
				else if(selectPiece && (float)(i+j)/2!=(i+j)/2 && toMove==black){
					//System.out.println("step b 2");
					
					selectCheck = false;
					end_i=i;
					end_j=j;
					
					if(start_i == end_i && start_j == end_j){
						//System.out.println("step b 2 same position i : "+i+" j : "+j);
					}
					else{
						//System.out.println("step b 2 i : "+i+" j : "+j);
						if (!incomplete){
							int[][] tmpBoard=new int[8][8];
							
							tmpBoard = Move.copyBoard(matrix);
							
							boardList.addElement(tmpBoard);
							
							if (!twoPlayer){
								sUndo.setVisible(true);
								sUndoDis.setVisible(false);
							}
							else{
								sUndo.setVisible(false);
								sUndoDis.setVisible(false);
							}
							
							int[] x = new int[16];
							for(int l=0;l<16;l++){
								x[l] = lastCap[l];
							}
							captureList.addElement(x);
							
							/*if (boardList.size()==1){
								int[] tmp = new int[16];
								captureList.addElement(tmp);
							}
							else{
								int[] tmp = new int[16];
								tmp = captureList.elementAt(boardList.size()-2);
								captureList.addElement(tmp);
								System.out.println("stepb--------");
								for(int k=0;k<captureList.size();k++){
									int[] tmpCapB = new int[16];
									tmpCapB = (int[])captureList.elementAt(k);
									for(int l=0;l<16;l++){
										System.out.print(tmpCapB[l]+"*");
									}
									System.out.println("stepb--------");
								}
								
							}*/
							
							/*for(int k=0;k<captureList.size();k++){
								int[] tmpCapB = new int[16];
								tmpCapB = (int[])captureList.elementAt(k);
								for(int l=0;l<16;l++){
									System.out.print(tmpCapB[l]+"*");
								}
								System.out.println("stepb--------");
							}*/

						}
						
						int status=Move.applyMove(matrix,start_i,start_j,end_i,end_j,true);
			
						switch(status){
						case Move.legalMove:
							//System.out.println("step b 2.1");
							incomplete=false;
							selectPiece=false;
							checkMove=1;
							blackMovePiece(start_i,start_j,end_i,end_j);
							blackClearPiece(start_i,start_j,end_i,end_j);
							break;
						case Move.illegalMove:
							//System.out.println("step b 2.1 ill");
							i = start_i;
							j = start_j;
							break;
						case Move.incompleteMove:
							//System.out.println("step b 2.3");
							incomplete=true;
							selectPiece=true;
							blackMovePiece(start_i,start_j,end_i,end_j);
							blackClearPiece(start_i,start_j,end_i,end_j);
							start_i=i;
							start_j=j;
						}
						if (!incomplete){
							
						}
						else{
							activePieceBlack(start_i,start_j);
						}
						if(Move.checkDraw(matrix)){
						//if (countDraw > 16){
							if (sound_enable){
								mWin.start();
							}
							resetButtonEnd();
							if (!twoPlayer){
								putStat("draw");
								sUndo.setVisible(false);
								sUndoDis.setVisible(true);
								mCurrentScene.unregisterTouchArea(sUndo);
							}
							mCurrentScene.registerTouchArea(bStart);
							mCurrentScene.attachChild(end);
							mCurrentScene.attachChild(draw);
							mCurrentScene.attachChild(bStart);
							mCurrentScene.registerTouchArea(sBack);
							mCurrentScene.attachChild(sBack);
							
							mCurrentScene.registerTouchArea(sAppStore);
							mCurrentScene.attachChild(sAppStore);
							startGame = false;
						}
						else{
								if(checkMove==1){
									
									checkMove=0;
									bTurn.setVisible(false);
									wTurn.setVisible(true);
									if (!twoPlayer){
										thinkThread = null;
										thinkThread = new Think2();
										thinkThread.start();
									}
									else{
										toMove= white;
										if(Move.noMoves(matrix,this.toMove)){
											if (sound_enable){
												mWin.start();
											}
											resetButtonEnd();
											mCurrentScene.registerTouchArea(bStart);
											mCurrentScene.attachChild(end);
											mCurrentScene.attachChild(humanwin);
											pieceWin = new Sprite(200, 390, this.itr_nBlack,
													getVertexBufferObjectManager()); 
											mCurrentScene.attachChild(pieceWin);
											mCurrentScene.attachChild(bStart);
											mCurrentScene.registerTouchArea(sBack);
											mCurrentScene.attachChild(sBack);
											
											mCurrentScene.registerTouchArea(sAppStore);
											mCurrentScene.attachChild(sAppStore);
											startGame = false;
										}
									}
									return false;
								}else if(checkMove==2){
									checkMove=0;
									toMove=black;
									if(Move.noMoves(matrix,this.toMove)){
										if (sound_enable){
											mLose.start();
										}
										resetButtonEnd();
										putStat("lose");
										mCurrentScene.registerTouchArea(bStart);
										mCurrentScene.attachChild(end);
										mCurrentScene.attachChild(lose);
										mCurrentScene.attachChild(bStart);
										mCurrentScene.registerTouchArea(sBack);
										mCurrentScene.attachChild(sBack);
										
										mCurrentScene.registerTouchArea(sAppStore);
										mCurrentScene.attachChild(sAppStore);
										
										sUndo.setVisible(false);
										sUndoDis.setVisible(true);
										mCurrentScene.unregisterTouchArea(sUndo);
										
										startGame = false;
									}
								}
							}
					}
				}
				else if (!incomplete && toMove==white && (matrix[i][j] == white || matrix[i][j] == whiteKing) && twoPlayer){

					if(selectCheck){
						if(!incomplete){
							if (start_i == i && start_j == j){
								//System.out.println("step b 1.1 same position");
							}
							else{
								selectPiece = true;
								start_i = i;
								start_j = j;
								if(! Move.canCapture(matrix,start_i,start_j) && Move.canCapture(matrix,toMove)){
									if(matrix[i][j] == 1){
										if (sound_enable){
											mDisable.start();
										}
										disablePieceWhite(i,j);
										return false;
									}
									else{
										if (sound_enable){
											mDisable.start();
										}
										disablePieceWhite(i,j);
										return false;
									}
								}
								else{
									if(matrix[i][j] == 1 || matrix[i][j] == 4){	
										activePieceWhite(i,j);
										return false;
									}
									else{
										activePieceWhite(i,j);
										return false;
									}				
								}
							}
						}	
						selectCheck = false;
					}
					else{
						
					}
				} 
				else if(selectPiece && (float)(i+j)/2!=(i+j)/2 && toMove==white ){
					if (incomplete){
						activePieceWhite(i,j);
					}
					selectCheck = false;
					end_i=i;
					end_j=j;
					
					if(start_i == end_i && start_j == end_j){
						//System.out.println("step b 2 same position i : "+i+" j : "+j);
					}
					else{
						int status=Move.applyMove(matrix,start_i,start_j,end_i,end_j,true);
			
						switch(status){
						case Move.legalMove:
							incomplete=false;
							selectPiece=false;
							checkMove=1;
							whiteMovePieceNormal(start_i,start_j,end_i,end_j);
							whiteClearPiece(start_i,start_j,end_i,end_j);
							break;
						case Move.illegalMove:
							//System.out.println("step b 2.1 ill");
							i = start_i;
							j = start_j;
							break;
						case Move.incompleteMove:
							incomplete=true;
							selectPiece=true;
							whiteMovePieceNormal(start_i,start_j,end_i,end_j);
							whiteClearPiece(start_i,start_j,end_i,end_j);
							start_i=i;
							start_j=j;
						}
						if (!incomplete){
							/*nBlackSelect.setVisible(false);
							nBlackKingSelect.setVisible(false);*/
						}
						else{
							//mDisable.start();
							activePieceWhite(start_i,start_j);
						}
						if(Move.checkDraw(matrix)){
						//if (countDraw > 16){
							if (sound_enable){
								mWin.start();
							}
							resetButtonEnd();
							mCurrentScene.registerTouchArea(bStart);
							mCurrentScene.attachChild(end);
							mCurrentScene.attachChild(draw);
							mCurrentScene.attachChild(bStart);
							mCurrentScene.registerTouchArea(sBack);
							mCurrentScene.attachChild(sBack);
							
							mCurrentScene.registerTouchArea(sAppStore);
							mCurrentScene.attachChild(sAppStore);
							startGame = false;
						}
						else{
							if(checkMove==1){
								checkMove=0;
								toMove=black;
								bTurn.setVisible(true);
								wTurn.setVisible(false);
								//return false;
							}else if(checkMove==2){//*/
								toMove=black;
								bTurn.setVisible(true);
								wTurn.setVisible(false);
							}
							if(Move.noMoves(matrix,this.toMove)){
								if (sound_enable){
									mWin.start();
								}
								resetButtonEnd();
								mCurrentScene.registerTouchArea(bStart);
								mCurrentScene.attachChild(end);
								pieceWin = new Sprite(200, 390, this.itr_nWhite,
										getVertexBufferObjectManager()); 
								mCurrentScene.attachChild(pieceWin);
								mCurrentScene.attachChild(humanwin);
								mCurrentScene.attachChild(bStart);
								mCurrentScene.registerTouchArea(sBack);
								mCurrentScene.attachChild(sBack);
								
								mCurrentScene.registerTouchArea(sAppStore);
								mCurrentScene.attachChild(sAppStore);
								startGame = false;
							}
						}
					}
				}	
			}//*/
		}
		return false;
	}

	@Override
	protected int getLayoutID() {
		// TODO Auto-generated method stub
		return R.layout.activity_main;
	}

	@Override
	protected int getRenderSurfaceViewID() {
		// TODO Auto-generated method stub
		return R.id.andengineID;
	}

	public void whiteMovePieceNormal(int pstart_i,int pstart_j,int pend_i,int pend_j){	
		
		for(int i=0;i<8;i++){
			if (wnPiece[i].posStartI == (pstart_i*100) && wnPiece[i].posStartJ == (pstart_j*100) ){
				if (sound_enable){
					mMove.start();
				}
				wnPiece[i].registerEntityModifier(new MoveModifier((float) 0.2, pstart_i*100+leftWidth , pend_i*100+leftWidth , pstart_j*100+topHeight ,pend_j*100+topHeight));
				wnPiece[i].setScale(1);
				
				if (pend_j == 7 && wnPiece[i].pieceType==white){
					//wnPiece[i].detachSelf();
					detach(wnPiece[i]);
					if (sound_enable){
						mMove.start();
					}
					wnPiece[i] = new Piece( pend_i*100+leftWidth, pend_j*100+topHeight , this.itr_nWhiteKing , getVertexBufferObjectManager()){
						public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
								float pTouchAreaLocalX, float pTouchAreaLocalY) {
							if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
								selectCheck = true;
							}
							return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
									pTouchAreaLocalY);
						}
					};

					mCurrentScene.attachChild(wnPiece[i]);
					mCurrentScene.registerTouchArea(wnPiece[i]);
					wnPiece[i].pieceType = whiteKing;
				}
				
				wnPiece[i].posStartI = pend_i*100;
				wnPiece[i].posStartJ = pend_j*100;
			}
		}
	}
	
	public void blackMovePiece(int pstart_i,int pstart_j,int pend_i,int pend_j){	
		
		for(int i=0;i<8;i++){
			if (bnPiece[i].posStartI == (pstart_i*100) && bnPiece[i].posStartJ == (pstart_j*100) ){
				if (sound_enable){
					mMove.start();
				}
				bnPiece[i].registerEntityModifier(new MoveModifier((float) 0.2, pstart_i*100+leftWidth , pend_i*100+leftWidth , pstart_j*100+topHeight ,pend_j*100+topHeight));
				bnPiece[i].setScale(1);
				
				if (pend_j == 0 && bnPiece[i].pieceType==black){
					//bnPiece[i].detachSelf();	
					detach(bnPiece[i]);
					bnPiece[i] = new Piece( pend_i*100+leftWidth, pend_j*100+topHeight , this.itr_nBlackKing , getVertexBufferObjectManager()){
						public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
								float pTouchAreaLocalX, float pTouchAreaLocalY) {
							if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
								selectCheck = true;
							}
							return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
									pTouchAreaLocalY);
						}
					};

					mCurrentScene.attachChild(bnPiece[i]);
					mCurrentScene.registerTouchArea(bnPiece[i]);
					bnPiece[i].pieceType = blackKing;
				}
				
				bnPiece[i].posStartI = pend_i*100;
				bnPiece[i].posStartJ = pend_j*100;
			}
		}
	}
	
	public void whiteMovePiece(int pstart_i,int pstart_j,int pend_i,int pend_j){	
		
		
		if(pend_i > 7 || pend_j > 7){
			// eat more 1 piece
			int startx=pstart_i;
			int starty=pstart_j;
			int endx=pend_i;
			int endy=pend_j;
			
			while(endx>0 || endy>0)
			{
				for(int i=0;i<8;i++){
					if (wnPiece[i].posStartI == (startx*100) && wnPiece[i].posStartJ == (starty*100) ){
						pause(300);
						if (sound_enable){
							mMove.start();	
						}
						wnPiece[i].registerEntityModifier(new MoveModifier((float) 0.3, startx*100+leftWidth , (endx%10)*100+leftWidth , starty*100+topHeight ,(endy%10)*100+topHeight));
								
						if ((endy%10) == 7 && wnPiece[i].pieceType==white){
									
							pause(300);
							//wnPiece[i].detachSelf();
							detach(wnPiece[i]);		
							wnPiece[i] = new Piece( (endx%10)*100+leftWidth, (endy%10)*100+topHeight , this.itr_nWhiteKing , getVertexBufferObjectManager());
		
							mCurrentScene.attachChild(wnPiece[i]);
							wnPiece[i].pieceType = whiteKing;
						}
								
						wnPiece[i].posStartI = (endx%10)*100;
						wnPiece[i].posStartJ = (endy%10)*100;
					}
				}
				pause(300);
				whiteClearPiece(startx, starty, endx%10, endy%10);
				pause(300);
				
				startx=endx%10;
				starty=endy%10;
				endx /=10;
				endy /=10; 
			}
		}
		else{
			for(int i=0;i<8;i++){
				if (wnPiece[i].posStartI == (pstart_i*100) && wnPiece[i].posStartJ == (pstart_j*100) ){
					pause(300);
					if (sound_enable){
						mMove.start();		
					}
					wnPiece[i].registerEntityModifier(new MoveModifier((float) 0.3, pstart_i*100+leftWidth , pend_i*100+leftWidth , pstart_j*100+topHeight ,pend_j*100+topHeight));
							
					if (pend_j == 7 && wnPiece[i].pieceType==white){
								
						pause(300);
						//wnPiece[i].detachSelf();
						detach(wnPiece[i]);		
						wnPiece[i] = new Piece( pend_i*100+leftWidth, pend_j*100+topHeight , this.itr_nWhiteKing , getVertexBufferObjectManager());
	
						mCurrentScene.attachChild(wnPiece[i]);
						wnPiece[i].pieceType = whiteKing;
					}
							
					wnPiece[i].posStartI = pend_i*100;
					wnPiece[i].posStartJ = pend_j*100;
				}
			}
			pause(300);
			whiteClearPiece(pstart_i, pstart_j, pend_i, pend_j);
		}
	}
	
	public void switchToMove(){
		int[] result=new int[4];
		if (bStart!=null){
			mCurrentScene.unregisterTouchArea(bStart);
		}
		if (sBack!=null){
			mCurrentScene.unregisterTouchArea(sBack);
		}
		if (sAppStore!=null){
			mCurrentScene.unregisterTouchArea(sAppStore);
		}
		if(toMove==black){
			toMove = white;
			
			if (level_select == 6){
				think.setVisible(true);
			}
			if (engine != null){
				engine = null;
			}
			engine = new Engine();
			//System.out.println("step back to switch : "+engine.stoprun);
			engine.MiniMax(matrix,0,level_select,result,toMove);
			//System.out.println("step back to switch");
			
			if (engine != null){
				if (!engine.stoprun){
				
					if (level_select == 6){
						think.setVisible(false);
					}
					if(result[0]==0 && result[1]==0){
						if (sound_enable){
							mWin.start();
						}
						/*if (engine != null){
							engine = null;
						}*/
						resetButtonEnd();
						putStat("win");
						mCurrentScene.registerTouchArea(bStart);
						mCurrentScene.attachChild(end);
						mCurrentScene.attachChild(win);
						mCurrentScene.attachChild(bStart);
						mCurrentScene.registerTouchArea(sBack);
						mCurrentScene.attachChild(sBack);
						
						mCurrentScene.registerTouchArea(sAppStore);
						mCurrentScene.attachChild(sAppStore);
						
						sUndo.setVisible(false);
						sUndoDis.setVisible(true);
						mCurrentScene.unregisterTouchArea(sUndo);
						
						startGame = false;
					}
					else{
						Move.moveBoard(matrix,result,true);
						
						whiteMovePiece(result[0],result[1],result[2],result[3]);
			
						checkMove=2;
						toMove=black;
						bTurn.setVisible(true);
						wTurn.setVisible(false);
						if(Move.noMoves(matrix,this.toMove)){
							if (sound_enable){
								mLose.start();
							}
							resetButtonEnd();
							putStat("lose");
							mCurrentScene.registerTouchArea(bStart);
							mCurrentScene.attachChild(end);
							mCurrentScene.attachChild(lose);
							mCurrentScene.attachChild(bStart);
							mCurrentScene.registerTouchArea(sBack);
							mCurrentScene.attachChild(sBack);
							
							mCurrentScene.registerTouchArea(sAppStore);
							mCurrentScene.attachChild(sAppStore);
							
							sUndo.setVisible(false);
							sUndoDis.setVisible(true);
							mCurrentScene.unregisterTouchArea(sUndo);
							
							startGame = false;
						}//*/
						/*if (engine != null){
							engine = null;
						}*/
					}
				}
				else{
					result[0] = 0;
					result[1] = 0;
					result[2] = 0;
					result[3] = 0;
					//System.out.println("step 1 stop back to switch");
				}
			}
			else{
				//System.out.println("step 2 stop back to switch");
			}
		}//*/
	}
	
	public void blackClearPiece(int pstart_i,int pstart_j,int pend_i,int pend_j){
		
		int posClearI = 0;
		int posClearJ = 0;
			if (Math.abs(pstart_i-pend_i) >= 2){

				if (pstart_i > pend_i){
					posClearI = pend_i + 1;
				}
				else{
					posClearI = pend_i - 1;
				}
				
				if (pstart_j > pend_j){
					posClearJ = pend_j + 1;
				}
				else{
					posClearJ = pend_j - 1;
				}				
				
				for(int i=0;i<8;i++){
					if (wnPiece[i].posStartI == (posClearI*100) && wnPiece[i].posStartJ == (posClearJ*100) ){
								whiteClear++;
								if (sound_enable){
									mEat.start();
								}
								wnPiece[i].setScale((float) 0.65);
								wnPiece[i].registerEntityModifier(new MoveModifier((float) 0.4, posClearI*100+leftWidth , (whiteClear-1)*57+10 , posClearJ*100+topHeight ,1075));
								wnPiece[i].posStartI = 0;
								wnPiece[i].posStartJ = 0;
								
								/*int[] tmpCap=new int[16];

								//System.out.println("stepx black cap--------------");
								
								tmpCap = (int[])captureList.elementAt(boardList.size()-1);
								for (int j=0;j<16;j++){
									if (tmpCap[j]==0){
										tmpCap[j] = wnPiece[i].pieceType;
										captureList.removeElementAt(boardList.size()-1);
										captureList.addElement(tmpCap);
										break;
									}
								}*/
								for (int j=0;j<16;j++){
									if (lastCap[j]==0){
										lastCap[j] = wnPiece[i].pieceType;
										break;
									}
								}
					}
				}
				
			}
	}	
	
	public void whiteClearPiece(int pstart_i,int pstart_j,int pend_i,int pend_j){
		int posClearI = 0;
		int posClearJ = 0;

			if (Math.abs(pstart_i-pend_i) >= 2){

				if (pstart_i > pend_i){
					posClearI = pend_i + 1;
				}
				else{
					posClearI = pend_i - 1;
				}
				if (pstart_j > pend_j){
					posClearJ = pend_j + 1;
				}
				else{
					posClearJ = pend_j - 1;
				}
				
				for(int i=0;i<8;i++){
					if (bnPiece[i].posStartI == (posClearI*100) && bnPiece[i].posStartJ == (posClearJ*100) ){
								
						blackClear++;
						if (sound_enable){
							mEat.start();
						}
						bnPiece[i].setScale((float) 0.65);
						bnPiece[i].registerEntityModifier(new MoveModifier((float) 0.3, posClearI*100+leftWidth, (blackClear-1)*57+10 , posClearJ*100+topHeight ,135));
						bnPiece[i].posStartI = 0;
						bnPiece[i].posStartJ = 0;
						
						//System.out.println("stepxx white cap--------------");
						
						for (int j=0;j<16;j++){
							if (lastCap[j]==0){
								lastCap[j] = bnPiece[i].pieceType;
								break;
							}
						}
						
						/*int[] tmpCap=new int[16];
						tmpCap = (int[])captureList.elementAt(captureList.size()-1);
						for (int j=0;j<16;j++){
							if (tmpCap[j]==0){
								tmpCap[j] = bnPiece[i].pieceType;
								int[] x = new int[16];
								for(int k =0;k<16;k++){
									x[k]=tmpCap[k];
								}
								captureList.removeElementAt(captureList.size()-1);
								captureList.addElement(x);
								break;
							}
						}	//*/
						
						
						//captureList.add(boardList.size()-1, tmpCap);
						/*int[] tmpCapB = new int[16];
						for(int k=0;k<captureList.size();k++){
							tmpCapB = (int[])captureList.elementAt(k);
							System.out.println(" step black cap : k : "+k);
							for(int j=0;j<16;j++){
								System.out.print(tmpCapB[j]+":");
							}
						}*/
					}
				}
			}
	}
	
	
	public void pause(long time){
		try{Thread.sleep(time);}
		catch(InterruptedException e){}
	}
	
	public void initMatrix(){
		for(int i=0;i<8;i++){
			for(int j=0;j<8;j++){
				if((float)(i+j)/2!=(i+j)/2){
					if(j<2){
						matrix[i][j] = white;
					}
					else if (j>5){
						matrix[i][j] = black;
					}
					else{
						matrix[i][j] = empty;
					}
				}
				else{
					matrix[i][j] = 9;
				}
			}
		}
	}
	
	public void initPiecePosition(){
		// white piece
		/*for(int i=0;i<8;i++){
			for(int j=0;j<2;j++){
				if((float)(i+j)/2!=(i+j)/2){
					
					if (wPiece[i][j] != null){
						wPiece[i][j].detachSelf();
					}
					
					wPiece[i][j] = new Piece( i*100+leftWidth , j*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager());
					wPiece[i][j].pieceType = white;
					wPiece[i][j].posStartI = i*100;
					wPiece[i][j].posStartJ = j*100;
					//wPiece[i][j].setVisible(false);
					wPiece[i][j].pieceType = white;
					mCurrentScene.attachChild(wPiece[i][j]);
				}
			}
		}*/
		/*System.out.println("step : "+(0/4));
		System.out.println("step : "+(0%4));
		
		System.out.println("step : "+(1/4));
		System.out.println("step : "+(1%4));
		
		System.out.println("step : "+(2/4));
		System.out.println("step : "+(2%4));
		
		System.out.println("step : "+(3/4));
		System.out.println("step : "+(3%4));
		
		System.out.println("step : "+(4/4));
		System.out.println("step : "+(4%4));
		
		System.out.println("step : "+(5/4));
		System.out.println("step : "+(5%4));
		
		System.out.println("step : "+(6/4));
		System.out.println("step : "+(6%4));
		
		System.out.println("step : "+(7/4));
		System.out.println("step : "+(7%4));//*/
		mCurrentScene.unregisterTouchArea(wnPiece[0]);
		if (wnPiece[0] != null){
			
			/*mEngine.runOnUpdateThread(new Runnable() {
				@Override
				public void run() {
					//mCurrentScene.detachChildren();
					wnPiece[0].detachSelf();

				}
			});*/
			detach(wnPiece[0]);
			//wnPiece[0].detachSelf();
		}
		wnPiece[0] = new Piece( 1*100+leftWidth , 0*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(wnPiece[1]);
		if (wnPiece[1] != null){
			//wnPiece[1].detachSelf();
			detach(wnPiece[1]);
		}
		wnPiece[1] = new Piece( 3*100+leftWidth , 0*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(wnPiece[2]);
		if (wnPiece[2] != null){
			//wnPiece[2].detachSelf();
			detach(wnPiece[2]);
		}
		wnPiece[2] = new Piece( 5*100+leftWidth , 0*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(wnPiece[3]);
		if (wnPiece[3] != null){
			//wnPiece[3].detachSelf();
			detach(wnPiece[3]);
		}
		wnPiece[3] = new Piece( 7*100+leftWidth , 0*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(wnPiece[4]);
		if (wnPiece[4] != null){
			//wnPiece[4].detachSelf();
			detach(wnPiece[4]);
		}
		wnPiece[4] = new Piece( 0*100+leftWidth , 1*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(wnPiece[5]);
		if (wnPiece[5] != null){
			//wnPiece[5].detachSelf();
			detach(wnPiece[5]);
		}
		wnPiece[5] = new Piece( 2*100+leftWidth , 1*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(wnPiece[6]);
		if (wnPiece[6] != null){
			//wnPiece[6].detachSelf();
			detach(wnPiece[6]);
		}
		wnPiece[6] = new Piece( 4*100+leftWidth , 1*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(wnPiece[7]);
		if (wnPiece[7] != null){
			//wnPiece[7].detachSelf();
			detach(wnPiece[7]);
		}
		wnPiece[7] = new Piece( 6*100+leftWidth , 1*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		for(int i=0;i<8;i++){
			wnPiece[i].pieceType = white;
			mCurrentScene.attachChild(wnPiece[i]);
			mCurrentScene.registerTouchArea(wnPiece[i]);	
		}
		
		mCurrentScene.unregisterTouchArea(bnPiece[0]);
		if (bnPiece[0] != null){
			//bnPiece[0].detachSelf();
			detach(bnPiece[0]);
		}
		bnPiece[0] = new Piece( 1*100+leftWidth , 6*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(bnPiece[1]);
		if (bnPiece[1] != null){
			//bnPiece[1].detachSelf();
			detach(bnPiece[1]);
		}
		bnPiece[1] = new Piece( 3*100+leftWidth , 6*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(bnPiece[2]);
		if (bnPiece[2] != null){
			//bnPiece[2].detachSelf();
			detach(bnPiece[2]);
		}
		bnPiece[2] = new Piece( 5*100+leftWidth , 6*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(bnPiece[3]);
		if (bnPiece[3] != null){
			//bnPiece[3].detachSelf();
			detach(bnPiece[3]);
		}
		bnPiece[3] = new Piece( 7*100+leftWidth , 6*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(bnPiece[4]);
		if (bnPiece[4] != null){
			//bnPiece[4].detachSelf();
			detach(bnPiece[4]);
		}
		bnPiece[4] = new Piece( 0*100+leftWidth , 7*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(bnPiece[5]);
		if (bnPiece[5] != null){
			//bnPiece[5].detachSelf();
			detach(bnPiece[5]);
		}
		bnPiece[5] = new Piece( 2*100+leftWidth , 7*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(bnPiece[6]);
		if (bnPiece[6] != null){
			//bnPiece[6].detachSelf();
			detach(bnPiece[6]);
		}
		bnPiece[6] = new Piece( 4*100+leftWidth , 7*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		mCurrentScene.unregisterTouchArea(bnPiece[7]);
		if (bnPiece[7] != null){
			//bnPiece[7].detachSelf();
			detach(bnPiece[7]);
		}
		bnPiece[7] = new Piece( 6*100+leftWidth , 7*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					selectCheck = true;
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		for(int i=0;i<8;i++){
			bnPiece[i].pieceType = black;
			mCurrentScene.attachChild(bnPiece[i]);
			mCurrentScene.registerTouchArea(bnPiece[i]);		
		}
		
		// black piece
		/*for(int i=0;i<8;i++){
			for(int j=0;j<2;j++){
				if((float)(i+j)/2!=(i+j)/2){
					
					if (bPiece[i][j] != null){
						bPiece[i][j].detachSelf();
					}
					
					bPiece[i][j] = new Piece( i*100+leftWidth , (j+6)*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
						public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
								float pTouchAreaLocalX, float pTouchAreaLocalY) {
							if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
								selectCheck = true;
							}
							return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
									pTouchAreaLocalY);
						}
					};
					bPiece[i][j].pieceType = black;
					bPiece[i][j].posStartI = i*100;
					bPiece[i][j].posStartJ = (j+6)*100;
					mCurrentScene.attachChild(bPiece[i][j]);
					//bPiece[i][j].setVisible(false);
					mCurrentScene.registerTouchArea(bPiece[i][j]);					
				}
			}
		}
		//*/
		/*System.out.println("step init");
		wPiece[0][1].setVisible(true);
		wPiece[0][1].setPosition(1*100 ,0*100+topHeight);
		wPiece[0][1].posStartI = 1*100;
		wPiece[0][1].posStartJ = 0*100;
		wPiece[0][1].setVisible(true);
		matrix[1][0] = white;
		
		wPiece[1][0].setVisible(true);
		wPiece[1][0].setPosition(4*100 ,1*100+topHeight);
		wPiece[1][0].posStartI = 4*100;
		wPiece[1][0].posStartJ = 1*100;
		wPiece[0][1].setVisible(true);
		matrix[4][1] = white;
		
		wPiece[2][1].setVisible(true);
		wPiece[2][1].setPosition(2*100 ,1*100+topHeight);
		wPiece[2][1].posStartI = 2*100;
		wPiece[2][1].posStartJ = 1*100;
		wPiece[0][1].setVisible(true);
		matrix[2][1] = white;
		
		wPiece[3][0].setVisible(true);
		wPiece[3][0].setPosition(4*100 ,5*100+topHeight);
		wPiece[3][0].posStartI = 4*100;
		wPiece[3][0].posStartJ = 5*100;
		wPiece[0][1].setVisible(true);
		matrix[4][5] = white;
		
		bPiece[0][1].setVisible(true);
		bPiece[0][1].setPosition(3*100 ,6*100+topHeight);
		bPiece[0][1].posStartI = 3*100;
		bPiece[0][1].posStartJ = 6*100;
		bPiece[0][1].setVisible(true);
		matrix[3][6] = black;
		
		bPiece[1][0].setVisible(true);
		bPiece[1][0].setPosition(5*100 ,4*100+topHeight);
		bPiece[1][0].posStartI = 5*100;
		bPiece[1][0].posStartJ = 4*100;
		bPiece[0][1].setVisible(true);
		matrix[5][4] = blackKing;
		
		bPiece[2][1].setVisible(true);
		bPiece[2][1].setPosition(2*100 ,3*100+topHeight);
		bPiece[2][1].posStartI = 2*100;
		bPiece[2][1].posStartJ = 3*100;
		bPiece[0][1].setVisible(true);
		matrix[2][3] = black;
		
		bPiece[3][0].setVisible(true);
		bPiece[3][0].setPosition(1*100 ,4*100+topHeight);
		bPiece[3][0].posStartI = 1*100;
		bPiece[3][0].posStartJ = 4*100;
		bPiece[0][1].setVisible(true);
		matrix[1][4] = black;
		
		bPiece[4][1].setVisible(true);
		bPiece[4][1].setPosition(2*100 ,7*100+topHeight);
		bPiece[4][1].posStartI = 2*100;
		bPiece[4][1].posStartJ = 7*100;
		bPiece[0][1].setVisible(true);
		matrix[2][7] = black;
		//*/
	}
	
	public void resetButtonEnd(){
		end = new Sprite(0, 0, this.itr_end,
				getVertexBufferObjectManager());
		end.setAlpha(80);
		win = new Sprite(60, 300, this.itr_win,
				getVertexBufferObjectManager()); 
		humanwin = new Sprite(60, 300, this.itr_humanwin,
				getVertexBufferObjectManager()); 
		lose = new Sprite(60, 300, this.itr_lose,
				getVertexBufferObjectManager()); 
		draw = new Sprite(60, 300, this.itr_draw,
				getVertexBufferObjectManager()); 
		                         //600
		bStart = new Sprite( 190 , 550 , itr_start , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					//System.out.println("step new");
					if (sound_enable){
						mMenu.start();
					}
					
					/*end.detachSelf();
					win.detachSelf();
					humanwin.detachSelf();
					lose.detachSelf();
					draw.detachSelf();*/
					
					detach(end);
					detach(win);
					detach(humanwin);
					detach(lose);
					detach(draw);
					
					if(!twoPlayer)
						mCurrentScene.registerTouchArea(sUndo);
					
					if (pieceWin != null){
						//pieceWin.detachSelf();
						detach(pieceWin);
					}
					/*sBack.detachSelf();
					this.detachSelf();*/
					detach(sBack);
					detach(sAppStore);
					detach(this);
					if (startTurn==black){
						startTurn=white;
						bTurn.setVisible(false);
						wTurn.setVisible(true);
						if (!twoPlayer){
							if (engine != null){
								engine = null;
							}

							initGame();
							toMove = black;

							if (thinkThread!=null){
								thinkThread = null;
							}
							thinkThread = new Think2();
							thinkThread.start();
						}
						else{
							if (engine != null){
								engine = null;
							}
							if (thinkThread!=null){
								thinkThread = null;
							}
							toMove = white;
							initGame();
						}
					}
					else{

						if (engine != null){
							//System.out.println("step 2 try to interrupt set null");
							if (!engine.stoprun){
								engine.stoprun = true;
								pause(1000);
							}
							engine = null;
						}

						initGame();
						startTurn=black;
						toMove = black;
						bTurn.setVisible(true);
						wTurn.setVisible(false);
					}//*/
					//mCurrentScene.unregisterTouchArea(bStart);
					//mCurrentScene.unregisterTouchArea(sBack);//*/
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};
		                        //880 
		sBack = new Sprite( 190 , 700 , itr_back , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					countDraw=0;
					if (sound_enable){
						mMenu.start();
					}
					boardList.removeAllElements();
					captureList.removeAllElements();

					//System.out.println("step back process");
					if (engine != null){
						//System.out.println("step back process engine");
						if (!engine.stoprun){
							engine.stoprun = true;
							pause(1000);
						}
						
						engine = null;
					}

					threadEnd=false;
					
					if (thinkThread!=null){
						thinkThread = null;
					}
					finish();
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};//*/
		
		
		sAppStore = new Sprite( 190 , 850 , itr_appStore , getVertexBufferObjectManager()){
			public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
					float pTouchAreaLocalX, float pTouchAreaLocalY) {
				
				if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
					countDraw=0;
					if (sound_enable){
						mMenu.start();
					}
					
					//System.out.println("----- App Store Process");
					adwall.loadAd();
					
					
					//finish();
				}
				return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
						pTouchAreaLocalY);
			}
		};//*/
	}
	
	public void activePieceBlack(int pstart_i,int pstart_j){
		for(int i=0;i<8;i++){
			if (bnPiece[i].posStartI == (pstart_i*100) && bnPiece[i].posStartJ == (pstart_j*100) ){
				bnPiece[i].setScale((float) 1.3);
				bnPiece[i].setAlpha((float) 1);
			}
			else if (bnPiece[i].posStartI != 0 || bnPiece[i].posStartJ != 0 ){
				bnPiece[i].setScale((float) 1);
				bnPiece[i].setAlpha((float) 1);
			}
		}
	}
	
	public void disablePieceBlack(int pstart_i,int pstart_j){
		for(int i=0;i<8;i++){
			if (bnPiece[i].posStartI == (pstart_i*100) && bnPiece[i].posStartJ == (pstart_j*100) ){
				bnPiece[i].setAlpha((float) 0.5);
				bnPiece[i].setScale((float) 1);
			}
			else if (bnPiece[i].posStartI != 0 || bnPiece[i].posStartJ != 0 ){
				bnPiece[i].setAlpha((float) 1);
				bnPiece[i].setScale((float) 1);
			}
		}
	}	
	
	public void activePieceWhite(int pstart_i,int pstart_j){
		for(int i=0;i<8;i++){
			if (wnPiece[i].posStartI == (pstart_i*100) && wnPiece[i].posStartJ == (pstart_j*100) ){
				wnPiece[i].setScale((float) 1.3);
				wnPiece[i].setAlpha((float) 1);
			}
			else if (wnPiece[i].posStartI != 0 || wnPiece[i].posStartJ != 0 ){
				wnPiece[i].setScale((float) 1);
				wnPiece[i].setAlpha((float) 1);
			}
		}
	}
	
	public void disablePieceWhite(int pstart_i,int pstart_j){
		for(int i=0;i<8;i++){
			if (wnPiece[i].posStartI == (pstart_i*100) && wnPiece[i].posStartJ == (pstart_j*100) ){
				wnPiece[i].setAlpha((float) 0.5);
				wnPiece[i].setScale((float) 1);
			}
			else if (wnPiece[i].posStartI != 0 || wnPiece[i].posStartJ != 0 ){
				wnPiece[i].setAlpha((float) 1);
				wnPiece[i].setScale((float) 1);
			}
		}
	}	
	
	protected void createMenuScene() {
		this.mMenuScene = new MenuScene(this.mCamera);

		final SpriteMenuItem resetMenuItem = new SpriteMenuItem(MENU_NEW, this.mMenuNewTextureRegion, this.getVertexBufferObjectManager());
		resetMenuItem.setBlendFunction(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
		this.mMenuScene.addMenuItem(resetMenuItem);

		final SpriteMenuItem quitMenuItem = new SpriteMenuItem(MENU_MAIN, this.mMenuMenuTextureRegion, this.getVertexBufferObjectManager());
		quitMenuItem.setBlendFunction(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
		this.mMenuScene.addMenuItem(quitMenuItem);

		this.mMenuScene.buildAnimations();

		this.mMenuScene.setBackgroundEnabled(false);

		this.mMenuScene.setOnMenuItemClickListener(this);
	}
	@Override
	public boolean onKeyDown(final int pKeyCode, final KeyEvent pEvent) {
		if(pKeyCode == KeyEvent.KEYCODE_MENU && pEvent.getAction() == KeyEvent.ACTION_DOWN) {
			if(this.mCurrentScene.hasChildScene()) {
				/* Remove the menu and reset it. */
				this.mMenuScene.back();
			} else {
				/* Attach the menu. */
				if(mMenuScene!=null)
					this.mCurrentScene.setChildScene(this.mMenuScene, false, true, true);
			}
			return true;
		} else {
			return super.onKeyDown(pKeyCode, pEvent);
		}
	}

	@Override
	public boolean onMenuItemClicked(final MenuScene pMenuScene, final IMenuItem pMenuItem, final float pMenuItemLocalX, final float pMenuItemLocalY) {
		if (sound_enable){
			mMenu.start();
		}
		switch(pMenuItem.getID()) {
			case MENU_NEW:
				/* Restart the animation. */
				this.mCurrentScene.reset();

				/* Remove the menu and reset it. */
				this.mCurrentScene.clearChildScene();
				this.mMenuScene.reset();
				boardList.removeAllElements();
				captureList.removeAllElements();

				think.setVisible(false);
				if (startTurn==black){
					startTurn=white;
					bTurn.setVisible(false);
					wTurn.setVisible(true);
					
					if (!twoPlayer){
						
						//System.out.println("step 1 try to interrupt");
						
						if (engine != null){
							//System.out.println("step 1 try to interrupt set null");
							if (!engine.stoprun){
								engine.stoprun = true;
								threadEnd = false;
								pause(1000);
								/*while (!threadEnd){
									System.out.println("step 1 null process");
								}*/
							}
							
							engine = null;
						}

						//System.out.println("step 1 try to interrupt success");
						initGame();
						toMove = black;

						if (thinkThread!=null){
							thinkThread = null;
						}
						thinkThread = new Think2();
						thinkThread.start();
					}
					else{
						initGame();
						toMove = white;
					}
				}
				else{
					startTurn=black;
					bTurn.setVisible(true);
					wTurn.setVisible(false);
					
					if (engine != null){
						//System.out.println("step 2 try to interrupt set null");
						if (!engine.stoprun){
							engine.stoprun = true;
							pause(1000);
							/*while (!threadEnd){
								System.out.println("step 2 null process");
							}*/
						}
						
						engine = null;
					}
					
					//System.out.println("step 2 try to interrupt success");
					initGame();//xx
					
					toMove = black;
					
					if (thinkThread!=null){
						thinkThread = null;
					}//*/
					
					
				}
				return true;
			case MENU_MAIN:
				boardList.removeAllElements();
				captureList.removeAllElements();

				countDraw=0;
				start_i= 0;
				start_j=0;
				end_i = 0;
				end_j = 0;
				if (engine != null){
					//System.out.println("step 3 try to interrupt set null");
					if (!engine.stoprun){
						engine.stoprun = true;
						pause(1000);
						/*while (!threadEnd){
							System.out.println("step 2 null process");
						}*/
					}
					
					engine = null;
				}
				//initGame();
				//System.out.println("step 3 init");
				this.finish();
				return true;
			default:
				countDraw=0;
				start_i= 0;
				start_j=0;
				end_i = 0;
				end_j = 0;
				return false;
		}
	}
	
	void initGame(){
		checkMove=0;
		initMatrix();
		initPiecePosition();
		startGame=true;
		start_i= 0;
		start_j=0;
		end_i = 0;
		end_j = 0;
		whiteClear=0;
		blackClear=0;
		countDraw=0;
		boardList.removeAllElements();
		captureList.removeAllElements();
		if (!twoPlayer){
			sUndoDis.setVisible(true);
			sUndo.setVisible(false);
			//mCurrentScene.registerTouchArea(sUndo);
		}
		else{
			sUndoDis.setVisible(false);
			sUndo.setVisible(false);
		}
		for(int i=0;i<16;i++)
			lastCap[i]=0;
	}
	
	public void printBoard(){
		int tempBoard[][] = new int[8][8];
		
		for (int k=boardList.size();k>0;k--){
			tempBoard = boardList.elementAt(k-1);
			for(int i=0;i<8;i++){
				System.out.println("   step board : ");
				for(int j=0;j<8;j++){
						System.out.print(tempBoard[i][j]+"-");
				}
			}
		}
	}
	
	public void undoLastTurn(){
		if (!twoPlayer){
			if (boardList.size()>0){
				if (toMove==white){
					bTurn.setVisible(true);
					wTurn.setVisible(false);
					if (engine != null){
						if (!engine.stoprun){
							engine.stoprun = true;
							pause(1000);
						}
						
						engine = null;
					}
	
					threadEnd=false;
					
					if (thinkThread!=null){
						thinkThread = null;
					}
					
					toMove = black;
					
					think.setVisible(false);
				}
				
				int tempBoard[][] = new int[8][8];
				int last = boardList.size();
				
				start_i=0;
				start_j=0;
				end_i=0;
				end_j=0;
				countDraw=0;
				
				tempBoard = boardList.elementAt(last-1);
		
				int bCount = 0;
				int wCount = 0;
				for(int i=0;i<8;i++){
					for(int j=0;j<8;j++){
						if((float)(i+j)/2!=(i+j)/2){
							if (tempBoard[i][j]==black){
								
								if (bnPiece[bCount] != null){
									mCurrentScene.unregisterTouchArea(bnPiece[bCount]);
									//bnPiece[bCount].detachSelf();
									detach(bnPiece[bCount]);
								}
								
								bnPiece[bCount] = new Piece( i*100+leftWidth , j*100+topHeight , this.itr_nBlack , getVertexBufferObjectManager()){
									public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
											float pTouchAreaLocalX, float pTouchAreaLocalY) {
										if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
											selectCheck = true;
										}
										return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
												pTouchAreaLocalY);
									}
								};
								
								bnPiece[bCount].pieceType = black;
								mCurrentScene.attachChild(bnPiece[bCount]);
								mCurrentScene.registerTouchArea(bnPiece[bCount]);		
								bCount++;
							}
							else if (tempBoard[i][j]==blackKing){
								
								if (bnPiece[bCount] != null){
									mCurrentScene.unregisterTouchArea(bnPiece[bCount]);
									//bnPiece[bCount].detachSelf();
									detach(bnPiece[bCount]);
								}
								
								bnPiece[bCount] = new Piece( i*100+leftWidth , j*100+topHeight , this.itr_nBlackKing , getVertexBufferObjectManager()){
									public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
											float pTouchAreaLocalX, float pTouchAreaLocalY) {
										if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
											selectCheck = true;
										}
										return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
												pTouchAreaLocalY);
									}
								};
								bnPiece[bCount].pieceType = blackKing;
								mCurrentScene.attachChild(bnPiece[bCount]);
								mCurrentScene.registerTouchArea(bnPiece[bCount]);
								bCount++;
							}
							else if (tempBoard[i][j]==white){
								
								if (wnPiece[wCount] != null){
									mCurrentScene.unregisterTouchArea(wnPiece[wCount]);
									//wnPiece[wCount].detachSelf();
									detach(wnPiece[wCount]);
								}
								
								wnPiece[wCount] = new Piece( i*100+leftWidth , j*100+topHeight , this.itr_nWhite , getVertexBufferObjectManager()){
									public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
											float pTouchAreaLocalX, float pTouchAreaLocalY) {
										if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
											selectCheck = true;
										}
										return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
												pTouchAreaLocalY);
									}
								};
								wnPiece[wCount].pieceType = white;
								mCurrentScene.attachChild(wnPiece[wCount]);
								mCurrentScene.registerTouchArea(wnPiece[wCount]);
								wCount++;
							}
							else if (tempBoard[i][j]==whiteKing){
								
								if (wnPiece[wCount] != null){
									mCurrentScene.unregisterTouchArea(wnPiece[wCount]);
									//wnPiece[wCount].detachSelf();
									detach(wnPiece[wCount]);
								}
								
								wnPiece[wCount] = new Piece( i*100+leftWidth , j*100+topHeight , this.itr_nWhiteKing , getVertexBufferObjectManager()){
									public boolean onAreaTouched(TouchEvent pSceneTouchEvent,
											float pTouchAreaLocalX, float pTouchAreaLocalY) {
										if (pSceneTouchEvent.getAction() == TouchEvent.ACTION_DOWN) {
											selectCheck = true;
										}
										return super.onAreaTouched(pSceneTouchEvent, pTouchAreaLocalX,
												pTouchAreaLocalY);
									}
								};
								wnPiece[wCount].pieceType = whiteKing;
								mCurrentScene.attachChild(wnPiece[wCount]);
								mCurrentScene.registerTouchArea(wnPiece[wCount]);
								wCount++;
							}//*/
						}
					}
				}
				
				matrix = Move.copyBoard(tempBoard);
				boardList.removeElementAt(last-1);
		
				int bCapCount = 1;
				int wCapCount = 1;
				int[] tmpCap = new int[16];
				tmpCap = (int[])captureList.elementAt(last-1);
				for(int j=0;j<16;j++){
					if (lastCap[j] != tmpCap[j]){	
						if (lastCap[j]==black || lastCap[j]==blackKing)
							blackClear--;
						else
							whiteClear--;
					}
					lastCap[j] = tmpCap[j];
				}
				for(int j=0;j<16;j++){
					if (tmpCap[j]==0){
						break;
					}
					else{
						if (tmpCap[j]==black){
							if (bnPiece[bCount] != null){
								//bnPiece[bCount].detachSelf();
								detach(bnPiece[bCount]);
							}
							//mCurrentScene.unregisterTouchArea(bnPiece[bCount]);			
							bnPiece[bCount] = new Piece( (bCapCount-1)*57+10 , 135 , this.itr_nBlack , getVertexBufferObjectManager()){};
							
							bnPiece[bCount].pieceType = black;
							bnPiece[bCount].setScale((float) 0.65);
							bnPiece[bCount].posStartI = 0;
							bnPiece[bCount].posStartJ = 0;
							mCurrentScene.attachChild(bnPiece[bCount]);//*/
							bCount++;					
							bCapCount++;
						}
						else if (tmpCap[j]==blackKing){
							if (bnPiece[bCount] != null){
								//bnPiece[bCount].detachSelf();
								detach(bnPiece[bCount]);
							}
							//mCurrentScene.unregisterTouchArea(bnPiece[bCount]);					
							bnPiece[bCount] = new Piece( (bCapCount-1)*57+10 , 135 , this.itr_nBlackKing , getVertexBufferObjectManager()){};
							
							bnPiece[bCount].pieceType = blackKing;
							bnPiece[bCount].setScale((float) 0.65);
							bnPiece[bCount].posStartI = 0;
							bnPiece[bCount].posStartJ = 0;
							mCurrentScene.attachChild(bnPiece[bCount]);
							bCount++;
							bCapCount++;
						}
						else if (tmpCap[j]==white){
							if (wnPiece[wCount] != null){
								//wnPiece[wCount].detachSelf();
								detach(wnPiece[wCount]);
							}
							//mCurrentScene.unregisterTouchArea(wnPiece[wCount]);				
							wnPiece[wCount] = new Piece( (wCapCount-1)*57+10 , 1075 , this.itr_nWhite , getVertexBufferObjectManager()){};
							
							wnPiece[wCount].pieceType = white;
							wnPiece[wCount].setScale((float) 0.65);
							wnPiece[wCount].posStartI = 0;
							wnPiece[wCount].posStartJ = 0;
							mCurrentScene.attachChild(wnPiece[wCount]);
							wCount++;
							wCapCount++;
						}
						else if (tmpCap[j]==whiteKing){
							if (wnPiece[wCount] != null){
								//wnPiece[wCount].detachSelf();
								detach(wnPiece[wCount]);
							}
							//mCurrentScene.unregisterTouchArea(wnPiece[wCount]);					
							wnPiece[wCount] = new Piece( (wCapCount-1)*57+10 , 1075 , this.itr_nWhiteKing , getVertexBufferObjectManager()){};
							
							wnPiece[wCount].pieceType = whiteKing;
							wnPiece[wCount].setScale((float) 0.65);
							wnPiece[wCount].posStartI = 0;
							wnPiece[wCount].posStartJ = 0;
							mCurrentScene.attachChild(wnPiece[wCount]);
							wCount++;
							wCapCount++;
						}//*/
					}
				}
		
				captureList.removeElementAt(last-1);
			}
			
			if (boardList.size()==0){
				sUndoDis.setVisible(true);
				sUndo.setVisible(false);
			}
		}
		
		/*for(int i=0;i<8;i++){
		System.out.println(" step matrix before : ");
		for(int j=0;j<8;j++){
				System.out.print(matrix[i][j]+"-");
		}
	}
	System.out.println(" step ------------- ");*/
		/*for (int k=boardList.size();k>0;k--){
		tempBoard = boardList.elementAt(k-1);
		for(int i=0;i<8;i++){
			System.out.println("   step board : ");
			for(int j=0;j<8;j++){
					System.out.print(tempBoard[i][j]+"-");
			}
		}
	}*/
	/*for(int k=0;k<captureList.size();k++){
	int[] tmpCapB = new int[16];
	tmpCapB = (int[])captureList.elementAt(k);
	System.out.println(" step black cap : k : "+k);
	for(int j=0;j<16;j++){
		System.out.print(tmpCapB[j]+":");
	}
	System.out.println(" step ----------------- ");
	}//*/
	
	/*tempBoard = boardList.elementAt(last-1);
	
	for(int i=0;i<8;i++){
		System.out.println(" step board : ");
		for(int j=0;j<8;j++){
				System.out.print(tempBoard[i][j]+"-");
		}
	}
	int[] tmpCapB = new int[16];
	tmpCapB = (int[])captureList.elementAt(last-1);
	System.out.println(" step cap : ");
	for(int j=0;j<16;j++){
		System.out.print(tmpCapB[j]+":");
	}
	System.out.println(" step ------------- ");*/
		
		/*for(int k=0;k<captureList.size();k++){
			int[] tmpCapB = new int[16];
			tmpCapB = (int[])captureList.elementAt(k);
			System.out.println(" step black cap : k bf : "+k);
			for(int j=0;j<16;j++){
				System.out.print(tmpCapB[j]+":");
			}
			System.out.println(" step ----------------- ");
		}*/
		

		/*for(int k=0;k<captureList.size();k++){
			int[] tmpCapB = new int[16];
			tmpCapB = (int[])captureList.elementAt(k);
			System.out.println(" step black cap af : k : "+k);
			for(int j=0;j<16;j++){
				System.out.print(tmpCapB[j]+":");
			}
			System.out.println(" step ----------------- ");
		}*/
		
		//System.out.println("step size of bl : "+boardList.size()+" cap : "+captureList.size());
		
		/*for(int i=0;i<8;i++){
			System.out.println(" step matrix after : ");
			for(int j=0;j<8;j++){
					System.out.print(matrix[i][j]+"-");
			}
		}*/
		//System.out.println(" step ------------- ");
		/*int[] tmpCapB = new int[16];
		tmpCapB = (int[])captureList.elementAt(last-1);
		System.out.println(" step cap : ");
		for(int j=0;j<16;j++){
			System.out.print(tmpCapB[j]+":");
		}*/

	}
	
	public void putStat(String str){
		SharedPreferences app_preferences = 
		        PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
		
		SharedPreferences.Editor editor = app_preferences.edit();
		if (level_select==2 && str=="win"){
			editor.putInt("l1win", app_preferences.getInt("l1win", 0)+1);
		}
		else if (level_select==2 && str=="draw"){
			editor.putInt("l1draw", app_preferences.getInt("l1draw", 0)+1);
		}
		else if (level_select==2 && str=="lose"){
			editor.putInt("l1lose", app_preferences.getInt("l1lose", 0)+1);
		}else if (level_select==4 && str=="win"){
			editor.putInt("l2win", app_preferences.getInt("l2win", 0)+1);
		}
		else if (level_select==4 && str=="draw"){
			editor.putInt("l2draw", app_preferences.getInt("l2draw", 0)+1);
		}
		else if (level_select==4 && str=="lose"){
			editor.putInt("l2lose", app_preferences.getInt("l2lose", 0)+1);
		}
		else if (level_select==6 && str=="win"){
			editor.putInt("l3win", app_preferences.getInt("l3win", 0)+1);
		}
		else if (level_select==6 && str=="draw"){
			editor.putInt("l3draw", app_preferences.getInt("l3draw", 0)+1);
		}
		else if (level_select==6 && str=="lose"){
			editor.putInt("l3lose", app_preferences.getInt("l3lose", 0)+1);
		}
		editor.commit();
	}
	
	void detach(final Sprite spr){
		mEngine.runOnUpdateThread(new Runnable() {
			@Override
			public void run() {
				//mCurrentScene.detachChildren();
				spr.detachSelf();
			}
		});
	}
}


