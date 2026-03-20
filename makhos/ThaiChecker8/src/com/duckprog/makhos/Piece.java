package com.duckprog.makhos;

import org.andengine.entity.sprite.Sprite;
//import org.andengine.input.touch.TouchEvent;
import org.andengine.opengl.texture.region.ITextureRegion;
import org.andengine.opengl.vbo.VertexBufferObjectManager;

public class Piece extends Sprite{
	// PieceType 1 = white , 2 = black , 0 = null
	public boolean selected = false;
	public int pieceType = 0;
	
	/*public int posI = 0;
	public int posJ = 0 ;*/
	
	public int startI = 0;
	public int startJ = 0;
	public int endI = 0;
	public int endJ = 0;
	
	public float posStartI = 0;
	public float posStartJ = 0;
	public float posEndI = 0;
	public float posEndJ = 0;
	
	public Piece(float pX, float pY, ITextureRegion pTextureRegion,
			VertexBufferObjectManager pVertexBufferObjectManager) {
		super(pX, pY, pTextureRegion, pVertexBufferObjectManager);
		// TODO Auto-generated constructor stub
		posStartI = pX-MainActivity.leftWidth;
		posStartJ = pY-MainActivity.topHeight;
	}
	
}
